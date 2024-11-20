package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"log"
	"os"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"github.com/joeychilson/infergo/models/yolo"
	"github.com/joeychilson/infergo/pkg/labels"
	"github.com/joeychilson/infergo/pkg/onnx"
	"github.com/joeychilson/infergo/pkg/postprocess"
	"github.com/joeychilson/infergo/pkg/preprocess"
)

var colors = []color.RGBA{
	{R: 255, G: 0, B: 0, A: 255},   // Red
	{R: 0, G: 255, B: 0, A: 255},   // Green
	{R: 0, G: 0, B: 255, A: 255},   // Blue
	{R: 255, G: 255, B: 0, A: 255}, // Yellow
	{R: 255, G: 0, B: 255, A: 255}, // Magenta
	{R: 0, G: 255, B: 255, A: 255}, // Cyan
}

func drawBoundingBox(img *image.RGBA, det postprocess.Detection, color color.RGBA) {
	for x := int(det.Box.X1); x <= int(det.Box.X2); x++ {
		img.Set(x, int(det.Box.Y1), color)
		img.Set(x, int(det.Box.Y2), color)
	}
	for y := int(det.Box.Y1); y <= int(det.Box.Y2); y++ {
		img.Set(int(det.Box.X1), y, color)
		img.Set(int(det.Box.X2), y, color)
	}

	label := fmt.Sprintf("%s (%.0f%%)", det.Label, det.Confidence*100)
	point := fixed.Point26_6{
		X: fixed.I(int(det.Box.X1)),
		Y: fixed.I(int(det.Box.Y1 - 5)),
	}

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color),
		Face: basicfont.Face7x13,
		Dot:  point,
	}
	d.DrawString(label)
}

func main() {
	modelPath := flag.String("model", "./.cache/models/yolos-small.onnx", "Path to YOLO ONNX model")
	imagePath := flag.String("image", "", "Path to input image")
	outputPath := flag.String("output", "output.png", "Path to save annotated image")
	confidenceThreshold := flag.Float64("confidence", 0.0, "Confidence threshold (0-1)")
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

	model, err := yolo.New(*modelPath)
	if err != nil {
		log.Fatalf("Failed to initialize model: %v", err)
	}
	defer model.Close()

	processedImg, err := preprocess.ProcessImage(img, preprocess.ProcessImageOptions{
		MinEdge:    800,
		MaxEdge:    1333,
		Mode:       preprocess.ResizeWithEdges,
		Mean:       [3]float32{0.485, 0.456, 0.406},
		StdDev:     [3]float32{0.229, 0.224, 0.225},
		CenterCrop: false,
	})
	if err != nil {
		log.Fatalf("Failed to preprocess image: %v", err)
	}

	output, err := model.Run(&yolo.Input{
		Height: processedImg.Height,
		Width:  processedImg.Width,
		Pixels: processedImg.Pixels,
	})
	if err != nil {
		log.Fatalf("Failed to run detection: %v", err)
	}

	detections, err := postprocess.ProcessDetections(
		output.Logits,
		output.Boxes,
		processedImg.OrigSize,
		postprocess.DetectionOptions{
			ConfThreshold: float32(*confidenceThreshold),
			IoUThreshold:  0.45,
			MaxDetections: 100,
			Labels:        labels.COCOLabels,
		},
	)
	if err != nil {
		log.Fatalf("Failed to process detections: %v", err)
	}

	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)

	for i, det := range detections {
		color := colors[i%len(colors)]
		drawBoundingBox(rgba, det, color)
	}

	outFile, err := os.Create(*outputPath)
	if err != nil {
		log.Fatalf("Failed to create output file: %v", err)
	}
	defer outFile.Close()

	if err := png.Encode(outFile, rgba); err != nil {
		log.Fatalf("Failed to encode output image: %v", err)
	}

	fmt.Printf("Found %d objects (confidence >= %.0f%%):\n", len(detections), *confidenceThreshold*100)
	fmt.Println("---------------------")
	for _, det := range detections {
		fmt.Printf("Object: %s\n", det.Label)
		fmt.Printf("Confidence: %.2f%%\n", det.Confidence*100)
		fmt.Printf("Bounding Box: (%.1f, %.1f) to (%.1f, %.1f)\n",
			det.Box.X1, det.Box.Y1, det.Box.X2, det.Box.Y2)
		fmt.Println("---------------------")
	}
	fmt.Printf("Annotated image saved to: %s\n", *outputPath)
}
