// dsv41probe: drive a dsv41 server exactly as a mainplane harness would (pkg/provider openai-chat envelope):
// compile a request from statefile records, send it, parse the stream, print records, stop and usage.
//   go run ./cmd/dsv41probe http://127.0.0.1:18099/v1 deepseek-v4.1-flash sess-0001
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/mainplane-ai/mainplane/pkg/provider"
	"github.com/mainplane-ai/mainplane/pkg/statefile"
)

func rec(kind statefile.Kind, body string, f func(*statefile.Record)) statefile.Record {
	r := statefile.Record{Header: statefile.Header{Kind: kind, Type: "text/plain"}, Body: []byte(body)}
	if f != nil {
		f(&r)
	}
	return r
}

func step(p provider.Provider, req provider.Request) {
	body, _, err := p.Envelope.Compile(req)
	if err != nil {
		fmt.Println("compile error:", err)
		os.Exit(1)
	}
	fmt.Println("--- request body:")
	fmt.Println(string(body))
	h, err := p.Step(context.Background(), req, func(r statefile.Record) {
		fmt.Printf("--- record kind=%s type=%s id=%q body=%q\n", r.Kind, r.Type, r.ID, string(r.Body))
	})
	if err != nil {
		fmt.Println("step error:", err)
		os.Exit(1)
	}
	u, _ := json.Marshal(h.Usage)
	fmt.Printf("--- step stop=%q provider=%s usage=%s\n", h.Stop, h.Provider, u)
}

func main() {
	url, model, key := os.Args[1], os.Args[2], os.Args[3]
	p := provider.OpenAIChat(url, "sk-local")
	tools := []provider.Tool{{Name: "run", Description: "Execute code on a worker",
		Schema: json.RawMessage(`{"type":"object","properties":{"worker":{"type":"string"},"code":{"type":"string"},"interpreter":{"type":"string"}},"required":["worker","code"]}`)}}
	ctx := []statefile.Record{
		rec(statefile.System, "You are a mainplane agent. Workers: admin.", nil),
		rec(statefile.System, "Session started 2026-10-30T00:00:00Z. Context limit 65536. Media input: none", func(r *statefile.Record) { r.Source = "session" }),
		rec(statefile.Message, "list the files", func(r *statefile.Record) { r.Via = "cli" }),
	}
	req := provider.Request{Model: model, Key: key, Tools: tools, Context: ctx,
		Params: map[string]json.RawMessage{"max_tokens": json.RawMessage(`64`), "temperature": json.RawMessage(`0`)}}
	fmt.Println("===== step 1 (first turn, cold)")
	step(p, req)
	// the tool loop: the assistant's call, its step, the tool result, then the next step with the SAME cache key
	call, _ := json.Marshal(provider.Call{Name: "run", Arguments: json.RawMessage(`{"worker":"admin","code":"ls"}`)})
	ctx = append(ctx,
		rec(statefile.Thinking, "let me check", func(r *statefile.Record) { r.Provider = "openai-chat"; r.ID = "a1" }),
		rec(statefile.Text, "I will list the files.", func(r *statefile.Record) { r.ID = "a2" }),
		statefile.Record{Header: statefile.Header{Kind: statefile.Call, Type: "application/json", ID: "call_fake1"}, Body: call},
		rec(statefile.Step, "", nil),
		rec(statefile.Result, "a.txt\nb.txt\n", func(r *statefile.Record) { r.For = "call_fake1"; r.Note = "time 2s context 120" }),
	)
	req.Context = ctx
	fmt.Println("===== step 2 (tool result, same prompt_cache_key)")
	step(p, req)
}
