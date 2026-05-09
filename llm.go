//go:build js && wasm
// +build js,wasm

package main

import (
	"encoding/binary"
	"fmt"
	"syscall/js"
	"time"
)

var (
	modelWeights []byte
	isLoaded     bool
)

// initModel receives the raw .safetensors ArrayBuffer from JavaScript
func initModel(this js.Value, args []js.Value) any {
	if len(args) < 1 {
		return "Error: No model buffer provided"
	}

	jsBuffer := args[0]
	length := jsBuffer.Get("length").Int()
	
	// Allocate Go memory and copy the weights from JS
	modelWeights = make([]byte, length)
	js.CopyBytesToGo(modelWeights, jsBuffer)
	isLoaded = true

	// Safetensors files start with an 8-byte little-endian uint64 indicating the JSON header size
	var headerSize uint64
	if length > 8 {
		headerSize = binary.LittleEndian.Uint64(modelWeights[:8])
	}

	successMsg := fmt.Sprintf("SmolLM2 (135M) loaded into Go memory!\nTotal size: %d bytes\nSafetensors header size: %d bytes", length, headerSize)
	fmt.Println(successMsg)
	return successMsg
}

// generateText simulates the inference pipeline using the loaded memory
func generateText(this js.Value, args []js.Value) any {
	prompt := args[0].String()
	
	promiseConstructor := js.Global().Get("Promise")
	return promiseConstructor.New(js.FuncOf(func(this js.Value, args []js.Value) any {
		resolve := args[0]
		
		go func() {
			if !isLoaded {
				resolve.Invoke("Error: Model weights not loaded yet.")
				return
			}

			// Simulate token generation latency 
			time.Sleep(1 * time.Second)
			
			// In a full implementation, you would multiply matrices here using the bytes in 'modelWeights'
			response := fmt.Sprintf("I am reading from %d bytes of model memory in Go.\n\nPrompt received: %s\n\n(Matrix multiplication/Transformer blocks go here for the next PR!)", len(modelWeights), prompt)
			
			resolve.Invoke(response)
		}()
		return nil
	}))
}

func main() {
	c := make(chan struct{})
	
	// Expose functions to JS window object
	js.Global().Set("initModel", js.FuncOf(initModel))
	js.Global().Set("generateText", js.FuncOf(generateText))
	
	fmt.Println("llm.go WASM initialized.")
	<-c
}