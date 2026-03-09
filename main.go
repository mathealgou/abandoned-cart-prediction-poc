package main

import (
	"fmt"
	"log"
	"main/api"
	"main/training"
	"net/http"
)

func main() {
	model := training.TrainModel()

	http.HandleFunc("/predict", api.NewPredictHandler(model))

	addr := ":8080"
	fmt.Printf("\nServer listening on %s\n", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
