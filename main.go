package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"mime"
	"path/filepath"

	"cloud.google.com/go/vertexai/genai"
)

// audioPrompt is a sample prompt type consisting of one audio asset, and a text question.
type audioPrompt struct {
	// audio is a Google Cloud Storage path starting with "gs://"
	audio string
	// question asked to the model
	question string
}

// transcribeAudio generates a response into w, based upon the prompt
// and audio provided.
// audio is a Google Cloud Storage path starting with "gs://"
func transcribeAudio(w io.Writer, prompt audioPrompt, projectID, location, modelName string) error {
	// prompt := audioPrompt{
	//      audio: "gs://cloud-samples-data/generative-ai/audio/pixel.mp3",
	//      question: `
	//              Can you transcribe this interview, in the format of timecode, speaker, caption.
	//              Use speaker A, speaker B, etc. to identify speakers.
	//      `,
	// },
	// location := "us-central1"
	// modelName := "gemini-1.5-flash-001"
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
