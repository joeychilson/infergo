package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"

	"github.com/joeychilson/infergo/models/resnet"
	"github.com/joeychilson/infergo/pkg/labels" // Make sure this import exists
	"github.com/joeychilson/infergo/pkg/onnx"
	"github.com/joeychilson/infergo/pkg/postprocess"
	"github.com/joeychilson/infergo/pkg/preprocess"
)

func main() {
	modelPath := flag.String("model", "./.cache/models/resnet50.onnx", "Path to ResNet ONNX model")
	imagePath := flag.String("image", "", "Path to input image")
	topK := flag.Int("top", 5, "Number of top predictions to show")
	confidenceThreshold := flag.Float64("confidence", 0.1, "Confidence threshold (0-1)")
	flag.Parse()

	if *imagePath == "" {
		log.Fatal("Please provide an image path using -image flag")
	}

	file, err := os.Open(*imagePath)
	if err != nil {
		log.Fatalf("Failed to open image: %v", err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		log.Fatalf("Failed to decode image: %v", err)
	}

	ctx := context.Background()

	runtime, err := onnx.New(ctx)
	if err != nil {
		log.Fatalf("Failed to create ONNX runtime: %v", err)
	}
	defer runtime.Close()

	model, err := resnet.New(*modelPath)
	if err != nil {
		log.Fatalf("Failed to initialize model: %v", err)
	}
	defer model.Close()

	processedImg, err := preprocess.ProcessImage(img, preprocess.ProcessImageOptions{
		Width:      224,
		Height:     224,
		Mode:       preprocess.ResizeAspectFill,
		Mean:       [3]float32{0.485, 0.456, 0.406},
		StdDev:     [3]float32{0.229, 0.224, 0.225},
		CenterCrop: true,
	})
	if err != nil {
		log.Fatalf("Failed to preprocess image: %v", err)
	}

	output, err := model.Run(&resnet.Input{Pixels: processedImg.Pixels})
	if err != nil {
		log.Fatalf("Failed to run inference: %v", err)
	}

	classifications, err := postprocess.ProcessClassification(output.Logits, postprocess.ClassificationOptions{
		Labels:   labels.ImageNetLabels,
		TopK:     *topK,
		MinScore: float32(*confidenceThreshold),
		Softmax:  true,
	})
	if err != nil {
		log.Fatalf("Failed to process results: %v", err)
	}

	fmt.Printf("\nTop %d predictions for %s:\n", *topK, *imagePath)
	fmt.Println("============================================")
	if len(classifications) == 0 {
		fmt.Printf("No predictions above %.1f%% confidence threshold\n", *confidenceThreshold*100)
	} else {
		for i, pred := range classifications {
			fmt.Printf("%d. %s\n", i+1, pred.Label)
			fmt.Printf("   Confidence: %.2f%%\n", pred.Confidence*100)
			fmt.Println("--------------------------------------------")
		}
	}
}
