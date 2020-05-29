package structures

import "time"

type ActivationData struct {
	Key          string `json:"key"`
	Dateexpiring string `json:"dateexpiring"`
	Datecreated  string `json:"datecreated"`
	Plan         int    `json:"plan"`
}

type Settings struct {
	ADMIN_KEY string `json:"adminkey"`
	API_KEY   string `json:"apikey"`
}

type Config struct {
	PLATFORM  string `json:"PLATFORM"`
	PROCESSOR string `json:"PROCESSOR"`
}

type ActivationSuccessResponse struct {
	Success      bool   `json:"success"`
	Expiringdate string `json:"expiringdate"`
	Daysleft     int    `json:"daysleft"`
	Plan         int    `json:"plan"`
}

type ActivationServerResponse struct {
	Responsecode        int    `json:"response_code"`
	Numberofactivations int    `json:"number_of_activations"`
	Plan                int    `json:"plan"`
	Datecreated         string `json:"date_created"`
	Dateexpiring        string `json:"date_expiring"`
}

type ModelInfo struct {
	Name        string    `json:"name"`
	Dateupdated time.Time `json:"dateupdated"`
	Modelsize   float32   `json:"size"`
}

type AllModels struct {
	Success bool        `json:"success"`
	Models  []ModelInfo `json:"models"`
}
