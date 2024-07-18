package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"path/filepath"

	"cloud.google.com/go/vertexai/genai"
)

func main() {
	buf := new(bytes.Buffer)
	prompt := audioPrompt{
		// audio: "gs://..",
		question: `
		Please provide a summary for the audio.
		Provide chapter titles with timestamps, be concise and short, no need to provide chapter summaries.
		Do not make up any information that is not part of the audio and do not be verbose.
	    `,
	}
	projectID := ""
	location := "us-central1"
	modelName := "gemini-1.5-flash-001"

	err := summarizeAudio(buf, prompt, projectID, location, modelName)
	if err != nil {
		log.Fatal(err)
		return
	}
}

// audioPrompt is a sample prompt type consisting of one audio asset, and a text question.
type audioPrompt struct {
	// audio is a Google Cloud Storage path starting with "gs://"
	audio string
	// question asked to the model
	question string
}

// summarizeAudio shows how to send an audio asset and a text question to a model, writing the response to the
// provided io.Writer.
// audio is a Google Cloud Storage path starting with "gs://"
func summarizeAudio(w io.Writer, prompt audioPrompt, projectID, location, modelName string) error {
	ctx := context.Background()

	client, err := genai.NewClient(ctx, projectID, location)
	if err != nil {
		return fmt.Errorf("unable to create client: %w", err)
	}
	defer client.Close()

	model := client.GenerativeModel(modelName)
	model.SetTemperature(0.4)

	// Given an audio file URL, prepare audio file as genai.Part
	part := genai.FileData{
		MIMEType: mime.TypeByExtension(filepath.Ext(prompt.audio)),
		FileURI:  prompt.audio,
	}

	res, err := model.GenerateContent(ctx, part, genai.Text(prompt.question))
	if err != nil {
		return fmt.Errorf("unable to generate contents: %w", err)
	}

	if len(res.Candidates) == 0 ||
		len(res.Candidates[0].Content.Parts) == 0 {
		return errors.New("empty response from model")
	}

	fmt.Fprintf(w, "generated summary:\n%s\n", res.Candidates[0].Content.Parts[0])
	return nil
}

// transcribeAudio generates a response into w, based upon the prompt
// and audio provided.
// audio is a Google Cloud Storage path starting with "gs://"
func transcribeAudio(w io.Writer, prompt audioPrompt, projectID, location, modelName string) error {
	ctx := context.Background()

	client, err := genai.NewClient(ctx, projectID, location)
	if err != nil {
		return fmt.Errorf("unable to create client: %w", err)
	}
	defer client.Close()

	model := client.GenerativeModel(modelName)

	// Optional: set an explicit temperature
	model.SetTemperature(0.4)

	// Given an audio file URL, prepare audio file as genai.Part
	img := genai.FileData{
		MIMEType: mime.TypeByExtension(filepath.Ext(prompt.audio)),
		FileURI:  prompt.audio,
	}

	res, err := model.GenerateContent(ctx, img, genai.Text(prompt.question))
	if err != nil {
		return fmt.Errorf("unable to generate contents: %w", err)
	}

	if len(res.Candidates) == 0 ||
		len(res.Candidates[0].Content.Parts) == 0 {
		return errors.New("empty response from model")
	}

	fmt.Fprintf(w, "generated transcript:\n%s\n", res.Candidates[0].Content.Parts[0])
	return nil
}
