package main

import (
	"net/http"

	"fyne.io/fyne/v2"
)

type Config struct {
	Backend          string `toml:"backend"` // "brave" or "claude"
	BraveAPIKey      string `toml:"brave_api_key"`
	ClaudeBaseURL    string `toml:"claude_base_url"`
	ClaudeModel      string `toml:"claude_model"`
	ClaudeAPIKey     string `toml:"claude_api_key"`
	ClaudeUseBedrock bool   `toml:"claude_use_bedrock"`
	SystemPrompt     string `toml:"system_prompt"`
	BuckyLib         string `toml:"bucky_lib"`
	WhisperModel     string `toml:"whisper_model"`
	PiperModel       string `toml:"piper_model"`
}

type keyBinding struct {
	key fyne.KeyName
	mod fyne.KeyModifier
}

type whiteDisabledTheme struct {
	fyne.Theme
}

type proxyTransport struct {
        inner      http.RoundTripper
        scheme     string
        host       string
        pathPrefix string
}

