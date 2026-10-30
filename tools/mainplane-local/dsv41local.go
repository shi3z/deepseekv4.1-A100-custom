// dsv41local: prepare a mainplane-server directory for a loopback-only run (no tunnel, no pointer):
// the harness key, network.json (name + device code), an api key named default, and print the tokens
// the CLI and the admin worker need.  Usage: dsv41local <dir> <network name>
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/mainplane-ai/mainplane/pkg/auth"
	"github.com/mainplane-ai/mainplane/pkg/pointer"
	"github.com/mainplane-ai/mainplane/pkg/server"
	"github.com/mainplane-ai/mainplane/pkg/worker"
)

func main() {
	dir, name := os.Args[1], os.Args[2]
	k, err := server.Key(dir)
	if err != nil {
		panic(err)
	}
	np := filepath.Join(dir, "network.json")
	var n struct {
		Name string `json:"name"`
		Code string `json:"code"`
	}
	if b, err := os.ReadFile(np); err == nil {
		if err := json.Unmarshal(b, &n); err != nil {
			panic(err)
		}
	} else if errors.Is(err, fs.ErrNotExist) {
		n.Name, n.Code = name, auth.NewCode()
		b, _ := json.Marshal(n)
		if err := os.WriteFile(np, b, 0o600); err != nil {
			panic(err)
		}
	} else {
		panic(err)
	}
	secret, err := server.Auth(dir).Issue(auth.Key, "default")
	if err != nil {
		panic(err)
	}
	pub := pointer.Encode(k)
	fmt.Printf("network %s %s\n", n.Name, n.Code)
	fmt.Printf("api_key %s\n", auth.Token(auth.Key, pub, secret))
	fmt.Printf("admin_token %s\n", auth.Token(auth.Join, pub, auth.Secret(k, worker.Admin+" "+n.Code)))
}
