package main

import (
	"fmt"
	"net/http"
	"bytes"
	"os"
	"os/exec"
	"strings"
	"sync/atomic"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
	"image/color"

	"github.com/BurntSushi/toml"
	"github.com/ardanlabs/bucky/pkg/whisper"
)

func appendMessage(history *widget.RichText, text string) {
	style := widget.RichTextStyle{}
	if strings.HasPrefix(text, "You: ") {
		style.ColorName = theme.ColorNameWarning // yellow
	}
	history.Segments = append(history.Segments, &widget.TextSegment{
		Text:  text,
		Style: style,
	})
	history.Refresh()
}

func cleanForSpeech(text string) string {
	if i := strings.Index(text, "<usage>"); i >= 0 {
		text = text[:i]
	}
	for {
		start := strings.Index(text, "```")
		if start < 0 {
			break
		}
		end := strings.Index(text[start+3:], "```")
		if end < 0 {
			text = text[:start]
			break
		}
		text = text[:start] + text[start+3+end+3:]
	}
	text = strings.ReplaceAll(text, "`", "")
	text = strings.ReplaceAll(text, "**", "")
	text = strings.ReplaceAll(text, "*", "")
	for strings.Contains(text, "\n\n") {
		text = strings.ReplaceAll(text, "\n\n", "\n")
	}
	return strings.TrimSpace(text)
}

func (t whiteDisabledTheme) Color(name fyne.ThemeColorName, variant fyne.ThemeVariant) color.Color {
	if name == theme.ColorNameDisabled {
		return color.White
	}
	return t.Theme.Color(name, variant)
}

func beep(freq int) {
	exec.Command("play", "-n", "-q", "-r", "22050", "-c", "1",
		"synth", "0.1", "sine", fmt.Sprintf("%d", freq)).Run()
}

func (t *proxyTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	r.URL.Scheme = t.scheme
	r.URL.Host = t.host
	r.URL.Path = t.pathPrefix + r.URL.Path
	r.Host = t.host
	return t.inner.RoundTrip(r)
}

func loadConfig() Config {
	var cfg Config
	path := os.ExpandEnv("$HOME/.agent-interface.toml")
	_, err := toml.DecodeFile(path, &cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v (expected at %s)\n", err, path)
		os.Exit(1)
	}
	return cfg
}

// --- Whisper transcription ---
func transcribe(rawData []byte) string {
        mu.Lock()
        defer mu.Unlock()
 
        numSamples := len(rawData) / 2 
        samples := make([]float32, numSamples)
        for i := 0; i < numSamples; i++ {
                s := int16(rawData[i*2]) | int16(rawData[i*2+1])<<8
                samples[i] = float32(s) / 32768.0
        }   
 
        wparams := whisper.FullDefaultParams(whisper.SamplingGreedy)
        wparams.NoTimestamps = 1 
        if err := whisper.Full(wctx, wparams, samples); err != nil {
                fmt.Fprintf(os.Stderr, "whisper.Full: %v\n", err)
                return ""
        }  
 
        var sb strings.Builder
        for i := int32(0); i < whisper.FullNSegments(wctx); i++ {
                sb.WriteString(whisper.FullGetSegmentText(wctx, i)) 
        }   
        return strings.TrimSpace(sb.String())
}
 
// --- TTS ---
func speak(text string) {
        cmd := exec.Command("piper", "--model", piperModel, "--output-raw")
        cmd.Stdin = strings.NewReader(text)
        var wavBuf bytes.Buffer
        cmd.Stdout = &wavBuf
        if err := cmd.Run(); err != nil {
                fmt.Fprintf(os.Stderr, "piper: %v\n", err)
                return
        }   
 
        aplayMu.Lock()
        aplayProc = playCmd()
        aplayProc.Stdin = &wavBuf
        if err := aplayProc.Start(); err != nil {
                aplayMu.Unlock()
                fmt.Fprintf(os.Stderr, "playback: %v\n", err)
                return
        }   
        atomic.StoreInt32(&aplayRunning, 1)
        aplayMu.Unlock()
 
        aplayProc.Wait()
        atomic.StoreInt32(&aplayRunning, 0)
}

