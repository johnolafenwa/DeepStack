package requests

type Request struct {
	Reqtype string `json:"reqtype"`
	Reqid   string `json:"reqid"`
}

type RecognitionRequest struct {
	Imgid   string `json:"imgid"`
	Reqtype string `json:"reqtype"`
	Reqid   string `json:"reqid"`
}

type FaceDetectionRequest struct {
	Imgid         string `json:"imgid"`
	Reqtype       string `json:"reqtype"`
	Reqid         string `json:"reqid"`
	Minconfidence string `json:"minconfidence"`
}

type FaceRecognitionRequest struct {
	Imgid         string `json:"imgid"`
	Reqtype       string `json:"reqtype"`
	Reqid         string `json:"reqid"`
	Minconfidence string `json:"minconfidence"`
}

type DetectionRequest struct {
	Imgid         string `json:"imgid"`
	Minconfidence string `json:"minconfidence"`
	Reqtype       string `json:"reqtype"`
	Reqid         string `json:"reqid"`
}

type FaceRegisterRequest struct {
	Userid  string   `json:"userid"`
	Images  []string `json:"images"`
	Reqid   string   `json:"reqid"`
	Reqtype string   `json:"reqtype"`
}

type FaceMatchRequest struct {
	Images  []string `json:"images"`
	Reqid   string   `json:"reqid"`
	Reqtype string   `json:"reqtype"`
}

type SuperresolutionRequest struct {
	Imgid   string `json:"imgid"`
	Reqtype string `json:"reqtype"`
	Reqid   string `json:"reqid"`
}

type FacelandmarkRequest struct {
	Imgid   string `json:"imgid"`
	Reqtype string `json:"reqtype"`
	Reqid   string `json:"reqid"`}