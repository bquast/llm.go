//go:build js && wasm
// +build js,wasm

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"syscall/js"
)

// Tensor Mapping
type TensorInfo struct {
	DataOffsets []uint64 `json:"data_offsets"`
}

type LlamaLayer struct {
	norm1, norm2           []float32
	wq, wk, wv, wo         []float32
	gate, up, down         []float32
}

var (
	modelWeights []byte
	tensors      map[string]TensorInfo
	
	// Model Architecture
	embed, norm_f, lm_head []float32
	layers                 []LlamaLayer
	kvCache                []float32 // Size: 30 * 2 * 2048 * 3 * 64 = 11,796,480 floats
	
	// Tokenizer
	vocab    map[string]int
	revVocab map[int]string
	isLoaded bool
)

// ----- NEURAL NETWORK MATH OPS -----

// rmsnorm computes Root Mean Square Normalization
func rmsnorm(out, x, weight []float32) {
	dim := len(x)
	var ss float32
	for i := 0; i < dim; i++ { ss += x[i] * x[i] }
	ss /= float32(dim)
	ss += 1e-5
	ss = float32(1.0 / math.Sqrt(float64(ss)))
	for i := 0; i < dim; i++ { out[i] = weight[i] * (ss * x[i]) }
}

// matmul performs Vector-Matrix Multiplication (y = Wx)
func matmul(out, x, w []float32) {
	d_out := len(out)
	d_in := len(x)
	for i := 0; i < d_out; i++ {
		var val float32
		row := w[i*d_in : (i+1)*d_in]
		for j := 0; j < d_in; j++ { val += row[j] * x[j] }
		out[i] = val
	}
}

// applyRoPE computes Rotary Positional Embeddings for Q and K
func applyRoPE(q, k []float32, pos, nHead, nKvHead, headDim int) {
	for i := 0; i < headDim; i += 2 {
		freq := 1.0 / math.Pow(10000.0, float64(i)/float64(headDim))
		val := float64(pos) * freq
		fcr, fci := float32(math.Cos(val)), float32(math.Sin(val))

		for h := 0; h < nHead; h++ {
			q0, q1 := q[h*headDim+i], q[h*headDim+i+1]
			q[h*headDim+i] = q0*fcr - q1*fci
			q[h*headDim+i+1] = q0*fci + q1*fcr
		}
		for h := 0; h < nKvHead; h++ {
			k0, k1 := k[h*headDim+i], k[h*headDim+i+1]
			k[h*headDim+i] = k0*fcr - k1*fci
			k[h*headDim+i+1] = k0*fci + k1*fcr
		}
	}
}

// swiglu computes the SwiGLU activation function
func swiglu(out, gate, up []float32) {
	for i := 0; i < len(out); i++ {
		x := gate[i]
		silu := x / (1.0 + float32(math.Exp(float64(-x))))
		out[i] = silu * up[i]
	}
}

// ----- THE FORWARD PASS -----

// forward runs a single token through all 30 LLaMA layers
func forward(pos, token int) int {
	dim := 576
	headDim := 64
	nHead := 9
	nKvHead := 3

	// 1. Embed Token
	x := make([]float32, dim)
	copy(x, embed[token*dim:(token+1)*dim])

	for l := 0; l < 30; l++ {
		// RMSNorm
		xb := make([]float32, dim)
		rmsnorm(xb, x, layers[l].norm1)

		// QKV Projections
		q := make([]float32, dim)
		k := make([]float32, 192)
		v := make([]float32, 192)
		matmul(q, xb, layers[l].wq)
		matmul(k, xb, layers[l].wk)
		matmul(v, xb, layers[l].wv)

		// RoPE
		applyRoPE(q, k, pos, nHead, nKvHead, headDim)

		// Grouped Query Attention (GQA)
		xb2 := make([]float32, dim)
		for h := 0; h < nHead; h++ {
			kv_h := h / 3
			
			// Store K & V in cache
			kIdx := l*786432 + 0*393216 + pos*192 + kv_h*64
			vIdx := l*786432 + 1*393216 + pos*192 + kv_h*64
			copy(kvCache[kIdx:kIdx+64], k[kv_h*64:(kv_h+1)*64])
			copy(kvCache[vIdx:vIdx+64], v[kv_h*64:(kv_h+1)*64])

			// Q @ K^T
			scores := make([]float32, pos+1)
			scale := float32(1.0 / math.Sqrt(64.0))
			qh := q[h*64 : (h+1)*64]
			for t := 0; t <= pos; t++ {
				var s float32
				kh := kvCache[l*786432 + t*192 + kv_h*64 : l*786432 + t*192 + kv_h*64 + 64]
				for i := 0; i < 64; i++ { s += qh[i] * kh[i] }
				scores[t] = s * scale
			}

			// Softmax
			maxS := scores[0]
			for t := 1; t <= pos; t++ { if scores[t] > maxS { maxS = scores[t] } }
			var sumS float32
			for t := 0; t <= pos; t++ {
				scores[t] = float32(math.Exp(float64(scores[t] - maxS)))
				sumS += scores[t]
			}
			for t := 0; t <= pos; t++ { scores[t] /= sumS }

			// Score @ V
			out_h := xb2[h*64 : (h+1)*64]
			for t := 0; t <= pos; t++ {
				vh := kvCache[l*786432 + 393216 + t*192 + kv_h*64 : l*786432 + 393216 + t*192 + kv_h*64 + 64]
				for i := 0; i < 64; i++ { out_h[i] += scores[t] * vh[i] }
			}
		}

		// Output Projection
		matmul(xb, xb2, layers[l].wo)
		for i := 0; i < dim; i++ { x[i] += xb[i] } // Residual

		// FFN (SwiGLU)
		rmsnorm(xb, x, layers[l].norm2)
		gate, up := make([]float32, 1536), make([]float32, 1536)
		matmul(gate, xb, layers[l].gate)
		matmul(up, xb, layers[l].up)
		swiglu(gate, gate, up)

		down := make([]float32, dim)
		matmul(down, gate, layers[l].down)
		for i := 0; i < dim; i++ { x[i] += down[i] } // Residual
	}

	// Final Norm & Classifier
	rmsnorm(x, x, norm_f)
	logits := make([]float32, 49152)
	matmul(logits, x, lm_head)

	// Argmax (Greedy Sampling)
	best := 0
	maxVal := float32(-1e9)
	for i, val := range logits {
		if val > maxVal {
			maxVal = val
			best = i
		}
	}
	return best
}

// ----- MEMORY SETUP -----

func getF32(key string) []float32 {
	t := tensors[key]
	data := modelWeights[t.DataOffsets[0]:t.DataOffsets[1]]
	f := make([]float32, len(data)/2)
	// BFloat16 to Float32 extraction
	for i := 0; i < len(f); i++ {
		u := uint32(data[i*2])<<16 | uint32(data[i*2+1])<<24
		f[i] = math.Float32frombits(u)
	}
	return f
}

func initModel(this js.Value, args []js.Value) any {
	jsBuffer, vocabStr := args[0], args[1].String()
	length := jsBuffer.Get("length").Int()

	modelWeights = make([]byte, length)
	js.CopyBytesToGo(modelWeights, jsBuffer)

	// Parse Safetensors Header
	headerSize := binary.LittleEndian.Uint64(modelWeights[:8])
	var rawHeader map[string]json.RawMessage
	json.Unmarshal(modelWeights[8:8+headerSize], &rawHeader)
	
	tensors = make(map[string]TensorInfo)
	for key, msg := range rawHeader {
		if key != "__metadata__" {
			var info TensorInfo
			json.Unmarshal(msg, &info)
			tensors[key] = info
		}
	}

	// Extract Tensors into Float32 Memory
	layers = make([]LlamaLayer, 30)
	for l := 0; l < 30; l++ {
		pfx := fmt.Sprintf("model.layers.%d.", l)
		layers[l].norm1 = getF32(pfx + "input_layernorm.weight")
		layers[l].wq = getF32(pfx + "self_attn.q_proj.weight")
		layers[l].wk = getF32(pfx + "self_attn.k_proj.weight")
		layers[l].wv = getF32(pfx + "self_attn.v_proj.weight")
		layers[l].wo = getF32(pfx + "self_attn.o_proj.weight")
		layers[l].norm2 = getF32(pfx + "post_attention_layernorm.weight")
		layers[l].gate = getF32(pfx + "mlp.gate_proj.weight")
		layers[l].up = getF32(pfx + "mlp.up_proj.weight")
		layers[l].down = getF32(pfx + "mlp.down_proj.weight")
	}
	embed = getF32("model.embed_tokens.weight")
	norm_f = getF32("model.norm.weight")
	lm_head = getF32("lm_head.weight")

	// Parse Vocab
	json.Unmarshal([]byte(vocabStr), &vocab)
	revVocab = make(map[int]string)
	for k, v := range vocab { revVocab[v] = k }

	// Init Cache
	kvCache = make([]float32, 11796480)
	isLoaded = true

	return fmt.Sprintf("Inference Engine Ready! 135M params mounted.")
}

// ----- TOKENIZATION & GENERATION BRIDGE -----

func generateText(this js.Value, args []js.Value) any {
	prompt := args[0].String()
	
	promiseConstructor := js.Global().Get("Promise")
	return promiseConstructor.New(js.FuncOf(func(this js.Value, args []js.Value) any {
		resolve := args[0]
		go func() {
			if !isLoaded { resolve.Invoke("Error: Model not loaded."); return }

			// 1. Tokenize Prompt (Longest Prefix Match)
			prompt = strings.ReplaceAll(prompt, " ", "Ġ")
			tokens := []int{}
			for len(prompt) > 0 {
				bestLen, bestId := 0, 0
				for word, id := range vocab {
					if strings.HasPrefix(prompt, word) && len(word) > bestLen {
						bestLen = len(word)
						bestId = id
					}
				}
				if bestLen == 0 { bestLen = 1 }
				tokens = append(tokens, bestId)
				prompt = prompt[bestLen:]
			}

			// 2. Process Prompt into KV Cache
			next := 0
			for i, t := range tokens { next = forward(i, t) }

			// 3. Auto-regressive Generation Loop
			for i := len(tokens); i < len(tokens)+30; i++ {
				// Decode and stream token to JS UI
				word := revVocab[next]
				word = strings.ReplaceAll(word, "Ġ", " ")
				word = strings.ReplaceAll(word, " ", " ")
				word = strings.ReplaceAll(word, "<|im_end|>", "\n")
				
				js.Global().Call("appendToken", word)

				// Predict next
				next = forward(i, next)
			}
			resolve.Invoke("")
		}()
		return nil
	}))
}

func main() {
	c := make(chan struct{})
	js.Global().Set("initModel", js.FuncOf(initModel))
	js.Global().Set("generateText", js.FuncOf(generateText))
	<-c
}