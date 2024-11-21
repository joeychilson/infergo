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

	text := "The [MASK] is a large animal that lives in the [MASK]."

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

	maskLogits, err := tok.MaskLogits(tokenOutput.Tokens, output.Logits)
	if err != nil {
		log.Fatal(err)
	}

	for _, ml := range maskLogits {
		classifictions, err := postprocess.ProcessClassification(ml.Logits, postprocess.ClassificationOptions{
			Labels:  tok.Labels(),
			TopK:    5,
			Softmax: true,
		})
		if err != nil {
			log.Fatal(err)
		}

		fmt.Println("\nTop 5 predictions for mask at position", ml.Position)
		for _, c := range classifictions {
			fmt.Printf("%s: %.2f\n", c.Label, c.Confidence*100)
		}
	}
}
