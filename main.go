package main

import (
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/driver/desktop"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	anthropicopt "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/ardanlabs/bucky/pkg/whisper"
	"github.com/aws/aws-sdk-go-v2/aws"
	openai "github.com/openai/openai-go/v3"
	openaiopt "github.com/openai/openai-go/v3/option"
)

var (
	appCfg          Config
	piperModel      string
	whisperModel    string
	buckyLib        string
	busy            int32
	wctx            whisper.Context
	mu              sync.Mutex
	aplayProc       *exec.Cmd
	aplayMu         sync.Mutex
	aplayRunning    int32
	currentShortcut *desktop.CustomShortcut

	braveClient openai.Client
	braveConvo  strings.Builder

	claudeClient   anthropic.Client
	claudeMessages []anthropic.MessageParam
	msgMu          sync.Mutex
)

func main() {
	appCfg = loadConfig()

	if appCfg.Backend == "" {
		appCfg.Backend = "brave"
	}
	if appCfg.ClaudeModel == "" {
		appCfg.ClaudeModel = "us.anthropic.claude-sonnet-4-6"
	}
	if appCfg.SystemPrompt == "" {
		appCfg.SystemPrompt = "You are a helpful voice assistant. Keep responses concise and conversational."
	}

	fmt.Fprintf(os.Stderr, "DEBUG: backend=%q model=%q lib=%q piper=%q\n",
		appCfg.Backend, appCfg.WhisperModel, appCfg.BuckyLib, appCfg.PiperModel)

	buckyLib = appCfg.BuckyLib
	whisperModel = appCfg.WhisperModel
	piperModel = appCfg.PiperModel

	// Init whisper
	if err := whisper.Load(buckyLib); err != nil {
		panic(err)
	}
	if err := whisper.Init(buckyLib); err != nil {
		panic(err)
	}
	cparams := whisper.ContextDefaultParams()
	var err error
	wctx, err = whisper.InitFromFileWithParams(whisperModel, cparams)
	if err != nil {
		panic(err)
	}
	defer whisper.Free(wctx)

	// Init LLM client
	switch appCfg.Backend {
	case "claude":
		var opts []anthropicopt.RequestOption
		if appCfg.ClaudeUseBedrock {
			token := appCfg.ClaudeAPIKey
			if token == "" {
				token = "-"
			}
			awsCfg := aws.Config{
				Region:                  "us-east-1",
				BearerAuthTokenProvider: bedrock.NewStaticBearerTokenProvider(token),
			}
			opts = append(opts, bedrock.WithConfig(awsCfg))
			if appCfg.ClaudeBaseURL != "" {
				parsed, err := url.Parse(appCfg.ClaudeBaseURL)
				if err != nil {
					fmt.Fprintf(os.Stderr, "invalid claude_base_url: %v\n", err)
					os.Exit(1)
				}
				transport := &proxyTransport{
					inner:      http.DefaultTransport,
					scheme:     parsed.Scheme,
					host:       parsed.Host,
					pathPrefix: strings.TrimRight(parsed.Path, "/"),
				}
				opts = append(opts, anthropicopt.WithHTTPClient(&http.Client{Transport: transport}))
			}
		} else {
			opts = append(opts, anthropicopt.WithAPIKey(appCfg.ClaudeAPIKey))
			if appCfg.ClaudeBaseURL != "" {
				opts = append(opts, anthropicopt.WithBaseURL(appCfg.ClaudeBaseURL))
			}
		}
		claudeClient = anthropic.NewClient(opts...)
	default:
		braveClient = openai.NewClient(
			openaiopt.WithAPIKey(appCfg.BraveAPIKey),
			openaiopt.WithBaseURL("https://api.search.brave.com/res/v1"),
		)
	}

	// --- Fyne app ---
	a := app.NewWithID("io.ktraister.agent")
	a.Settings().SetTheme(theme.DarkTheme())
	w := a.NewWindow("Agent Interface")
	w.Resize(fyne.NewSize(600, 500))

	// Text input
	textInput := widget.NewEntry()
	textInput.PlaceHolder = "Or type here..."

	// Conversation history
	history := widget.NewRichText(
		&widget.TextSegment{
			Text: "",
			Style: widget.RichTextStyle{
				Alignment: fyne.TextAlignLeading,
			},
		},
	)
	historyScroll := container.NewScroll(history)

	// Status label
	status := widget.NewLabel("Ready")

	// Voice mode toggle
	voiceCheck := widget.NewCheck("Voice response", func(checked bool) {})
	voiceCheck.SetChecked(true)

	// Send button
	sendBtn := widget.NewButton("Send", func() {
		q := strings.TrimSpace(textInput.Text)
		if q == "" {
			return
		}
		textInput.SetText("")
		go handleQuery(q, history, status, voiceCheck)
	})
	sendBtn.Importance = widget.HighImportance

	// Clear Button
	clearBtn := widget.NewButton("Clear", func() {
		history.Segments = append(history.Segments, &widget.TextSegment{
			Text:  "",
			Style: widget.RichTextStyle{},
		})
		history.Refresh()
		braveConvo.Reset()
		msgMu.Lock()
		claudeMessages = nil
		msgMu.Unlock()
	})
	clearBtn.Importance = widget.WarningImportance

	// Talk button
	var talkBtn *widget.Button

	triggerTalk := func() {
		if atomic.LoadInt32(&busy) == 1 {
			status.SetText("Busy...")
			return
		}
		atomic.StoreInt32(&busy, 1)

		talkBtn.Disable()
		status.SetText("Recording... speak now")
		beep(800)

		go func() {
			wavData := recordAudio()
			beep(400)
			if wavData == nil {
				atomic.StoreInt32(&busy, 0)
				fyne.DoAndWait(func() {
					status.SetText("Ready")
					talkBtn.Enable()
				})
				return
			}
			fyne.DoAndWait(func() { status.SetText("Transcribing...") })
			text := transcribe(wavData)
			if text == "" {
				atomic.StoreInt32(&busy, 0)
				fyne.DoAndWait(func() {
					status.SetText("Ready")
					talkBtn.Enable()
				})
				return
			}
			fyne.DoAndWait(func() { talkBtn.Enable() })

			go func() {
				handleQuery(text, history, status, voiceCheck)
				atomic.StoreInt32(&busy, 0)
			}()
		}()
	}

	talkBtn = widget.NewButton("Talk", triggerTalk)
	talkBtn.Importance = widget.SuccessImportance

	// Stop button
	stopBtn := widget.NewButton("Stop", func() {
		aplayMu.Lock()
		if aplayProc != nil && atomic.LoadInt32(&aplayRunning) == 1 {
			aplayProc.Process.Kill()
		}
		aplayMu.Unlock()
		status.SetText("Ready")
	})
	stopBtn.Importance = widget.DangerImportance

	// Configure button
	configureBtn := widget.NewButton("Configure", func() {
		var options []string
		keyMap := make(map[string]keyBinding)

		if runtime.GOOS == "darwin" {
			options = []string{
				"Cmd+Space",
				"Cmd+T",
				"Cmd+Shift+Space",
				"F9",
				"F10",
			}
			keyMap["Cmd+Space"] = keyBinding{fyne.KeySpace, fyne.KeyModifierSuper}
			keyMap["Cmd+T"] = keyBinding{fyne.KeyT, fyne.KeyModifierSuper}
			keyMap["Cmd+Shift+Space"] = keyBinding{fyne.KeySpace, fyne.KeyModifierSuper | fyne.KeyModifierShift}
			keyMap["F9"] = keyBinding{fyne.KeyF9, 0}
			keyMap["F10"] = keyBinding{fyne.KeyF10, 0}
		} else {
			options = []string{
				"Ctrl+Space",
				"Ctrl+T",
				"Delete",
				"F9",
				"F10",
				"Ctrl+Alt+Space",
			}
			keyMap["Ctrl+Space"] = keyBinding{fyne.KeySpace, fyne.KeyModifierControl}
			keyMap["Ctrl+T"] = keyBinding{fyne.KeyT, fyne.KeyModifierControl}
			keyMap["Delete"] = keyBinding{fyne.KeyDelete, 0}
			keyMap["F9"] = keyBinding{fyne.KeyF9, 0}
			keyMap["F10"] = keyBinding{fyne.KeyF10, 0}
			keyMap["Ctrl+Alt+Space"] = keyBinding{fyne.KeySpace, fyne.KeyModifierControl | fyne.KeyModifierAlt}
		}

		keySelect := widget.NewSelect(options, func(selected string) {
			if currentShortcut != nil {
				w.Canvas().RemoveShortcut(currentShortcut)
			}
			binding := keyMap[selected]
			currentShortcut = &desktop.CustomShortcut{KeyName: binding.key, Modifier: binding.mod}
			w.Canvas().AddShortcut(currentShortcut, func(_ fyne.Shortcut) {
				triggerTalk()
			})
			status.SetText("Hotkey set: " + selected)
		})

		content := container.NewVBox(
			widget.NewLabel("Choose a hotkey:"),
			keySelect,
		)

		dlg := dialog.NewCustom("Set Talk Hotkey", "Close", content, w)
		dlg.Resize(fyne.NewSize(300, 120))
		dlg.Show()
	})

	// Layout
	buttonRow := container.NewHBox(talkBtn, stopBtn, clearBtn, configureBtn, voiceCheck)
	inputRow := container.NewBorder(nil, nil, nil, sendBtn, textInput)

	w.SetContent(container.NewBorder(
		container.NewVBox(buttonRow, status), // top
		inputRow,                             // bottom
		nil, nil,                             // left, right
		historyScroll,
	))

	// Default hotkey: Cmd+Space on macOS, Ctrl+Space on Linux
	if runtime.GOOS == "darwin" {
		currentShortcut = &desktop.CustomShortcut{
			KeyName:  fyne.KeySpace,
			Modifier: fyne.KeyModifierSuper,
		}
	} else {
		currentShortcut = &desktop.CustomShortcut{
			KeyName:  fyne.KeySpace,
			Modifier: fyne.KeyModifierControl,
		}
	}
	w.Canvas().AddShortcut(currentShortcut, func(_ fyne.Shortcut) {
		triggerTalk()
	})
	if runtime.GOOS == "darwin" {
		status.SetText("Ready (hotkey: Cmd+Space)")
	} else {
		status.SetText("Ready (hotkey: Ctrl+Space)")
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGUSR1)
	go func() {
		for range sigCh {
			fyne.DoAndWait(func() {
				triggerTalk()
			})
		}
	}()

	w.ShowAndRun()
}

// --- Handle a query (dispatches to configured backend) ---
func handleQuery(query string, history *widget.RichText, status *widget.Label, voiceCheck *widget.Check) {
	if query == "[ Silence ]" {
		return
	}

	fyne.DoAndWait(func() {
		appendMessage(history, "You: "+query)
		status.SetText("Thinking...")
	})

	var dirtyAnswer string
	var err error

	switch appCfg.Backend {
	case "claude":
		dirtyAnswer, err = queryBedrock(query)
	default:
		dirtyAnswer, err = queryBrave(query)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "LLM error: %v\n", err)
		fyne.DoAndWait(func() { status.SetText("Error: " + err.Error()) })
		return
	}

	fullAnswer := cleanForSpeech(dirtyAnswer)

	fyne.DoAndWait(func() {
		appendMessage(history, "AI: "+fullAnswer)
		appendMessage(history, "")
	})

	if voiceCheck.Checked {
		fyne.DoAndWait(func() { status.SetText("Speaking...") })
		speak(fullAnswer)
		fyne.DoAndWait(func() { status.SetText("Ready") })
	} else {
		fyne.DoAndWait(func() { status.SetText("Ready") })
	}
}

func queryBrave(query string) (string, error) {
	braveConvo.WriteString("User: " + query + "\n")

	contextPrefix := ""
	if braveConvo.Len() > 0 {
		contextPrefix = "Previous conversation:\n" + braveConvo.String() + "\n\nCurrent question: "
	}

	stream := braveClient.Chat.Completions.NewStreaming(context.Background(), openai.ChatCompletionNewParams{
		Model: openai.ChatModel("brave"),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(contextPrefix + query),
		},
	})

	var answer strings.Builder
	for stream.Next() {
		chunk := stream.Current()
		answer.WriteString(chunk.Choices[0].Delta.Content)
	}
	if err := stream.Err(); err != nil {
		return "", err
	}

	fullAnswer := answer.String()
	braveConvo.WriteString("AI: " + fullAnswer + "\n")
	return fullAnswer, nil
}

func queryBedrock(query string) (string, error) {
	msgMu.Lock()
	claudeMessages = append(claudeMessages, anthropic.NewUserMessage(anthropic.NewTextBlock(query)))
	messages := make([]anthropic.MessageParam, len(claudeMessages))
	copy(messages, claudeMessages)
	msgMu.Unlock()

	stream := claudeClient.Messages.NewStreaming(context.Background(), anthropic.MessageNewParams{
		Model:     appCfg.ClaudeModel,
		MaxTokens: 4096,
		System: []anthropic.TextBlockParam{{
			Text: appCfg.SystemPrompt,
		}},
		Messages: messages,
	})

	accumulated := anthropic.Message{}
	for stream.Next() {
		accumulated.Accumulate(stream.Current())
	}
	if err := stream.Err(); err != nil {
		return "", err
	}

	var answer strings.Builder
	for _, block := range accumulated.Content {
		switch v := block.AsAny().(type) {
		case anthropic.TextBlock:
			answer.WriteString(v.Text)
		}
	}

	fullAnswer := answer.String()

	msgMu.Lock()
	claudeMessages = append(claudeMessages, anthropic.NewAssistantMessage(anthropic.NewTextBlock(fullAnswer)))
	msgMu.Unlock()

	return fullAnswer, nil
}

// --- Audio recording (silence detection) ---
func recordAudio() []byte {
	const (
		sampleRate     = 16000
		frameMs        = 30
		frameSamples   = sampleRate * frameMs / 1000 // 480
		speechThresh   = 0.015                       // RMS threshold (tune: raise if noisy, lower if quiet)
		silenceMs      = 1000                        // stop after 1s of silence
		maxDurationMs  = 20000                       // hard cap: 20s
		startTimeoutMs = 3000                        // give up if no speech within 3s of starting
	)

	tmpFile, err := os.CreateTemp("", "rec_*.raw")
	if err != nil {
		return nil
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	cmd := recordCmd()
	pipe, err := cmd.StdoutPipe()
	if err != nil {
		tmpFile.Close()
		return nil
	}
	if err := cmd.Start(); err != nil {
		tmpFile.Close()
		return nil
	}

	frameBytes := frameSamples * 2 // S16 = 2 bytes per sample
	frame := make([]byte, frameBytes)

	var (
		silenceFrames  int
		totalFrames    int
		speechDetected bool
		maxFrames      = maxDurationMs / frameMs
		startTimeout   = startTimeoutMs / frameMs
	)

	for totalFrames < maxFrames {
		_, err := io.ReadFull(pipe, frame)
		if err != nil {
			break
		}
		totalFrames++

		// Compute RMS
		rms := 0.0
		for i := 0; i < frameSamples; i++ {
			s := int16(frame[i*2]) | int16(frame[i*2+1])<<8
			f := float64(s) / 32768.0
			rms += f * f
		}
		rms = math.Sqrt(rms / float64(frameSamples))

		isSpeech := rms > speechThresh

		if isSpeech {
			speechDetected = true
			silenceFrames = 0
			tmpFile.Write(frame)
		} else if speechDetected {
			tmpFile.Write(frame)
			silenceFrames++
			if silenceFrames*frameMs >= silenceMs {
				break
			}
		} else {
			if totalFrames >= startTimeout {
				break
			}
		}
	}

	cmd.Process.Kill()
	cmd.Wait()
	tmpFile.Close()

	data, err := os.ReadFile(tmpPath)
	if err != nil || len(data) < 100 {
		return nil
	}
	return data
}
