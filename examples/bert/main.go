package main

import (
	"context"
	"fmt"
	"log"

	"github.com/joeychilson/infergo/models/bert"
	"github.com/joeychilson/infergo/pkg/onnx"
	"github.com/joeychilson/infergo/pkg/postprocess"
	"github.com/joeychilson/infergo/pkg/tokenizer"
)

func main() {
	runtime, err := onnx.New(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	defer runtime.Close()

	tok, err := tokenizer.NewBERTTokenizer()
	if err != nil {
		log.Fatal(err)
	}

	model, err := bert.New(".cache/models/distilbert.onnx")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	text := "[mask] is the capital of France."

	tokenOutput, err := tok.Encode(text, 512)
	if err != nil {
		log.Fatal(err)
	}

	output, err := model.Run(&bert.Input{
		InputIds:      tokenOutput.InputIds,
		AttentionMask: tokenOutput.AttentionMask,
	})
	if err != nil {
		log.Fatal(err)
	}

	maskPosition := tok.MaskPosition(tokenOutput.Tokens)
	vocabSize := tok.VocabSize()

	maskLogits := output.Logits[maskPosition*vocabSize : (maskPosition+1)*vocabSize]

	classifictions, err := postprocess.ProcessClassification(maskLogits, postprocess.ClassificationOptions{
		Labels:  tok.Labels(),
		TopK:    5,
		Softmax: true,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("\nTop 5 predictions:")
	for _, c := range classifictions {
		fmt.Printf("%s: %.2f\n", c.Label, c.Confidence*100)
	}
}
