package main

import (
	"archive/zip"
	"context"
	"encoding/json"
	"flag"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"os/signal"
	"path"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis"
	uuid "github.com/satori/go.uuid"

	"database/sql"
	"fmt"
	"path/filepath"

	"deepstack.io/server/middlewares"
	"deepstack.io/server/requests"
	"deepstack.io/server/response"
	"deepstack.io/server/structures"
	"deepstack.io/server/utils"
	_ "github.com/mattn/go-sqlite3"
)

var temp_path = "/deeptemp/"
var DATA_DIR = "/datastore"

var db *sql.DB

var API_KEY = ""

var sub_key = ""

var state = true
var gpu = true
var request_timeout = 60.0

var expiring_date = time.Now()

var settings structures.Settings
var sub_data = structures.ActivationData{}
var redis_client *redis.Client

func scene(c *gin.Context) {

	img_id := uuid.NewV4().String()
	req_id := uuid.NewV4().String()

	file, _ := c.FormFile("image")

	c.SaveUploadedFile(file, filepath.Join(temp_path, img_id))

	req_data := requests.RecognitionRequest{Imgid: img_id, Reqid: req_id, Reqtype: "scene"}
	req_string, _ := json.Marshal(req_data)

	redis_client.RPush("scene_queue", req_string)

	t1 := time.Now()

	for true {

		output, _ := redis_client.Get(req_id).Result()
		duration := time.Since(t1).Seconds()

		if output != "" {

			var res response.RecognitionResponse
			json.Unmarshal([]byte(output), &res)

			if res.Success == false {

				var error_response response.ErrorResponseInternal

				json.Unmarshal([]byte(output), &error_response)

				final_res := response.ErrorResponse{Success: false, Error: error_response.Error}

				c.JSON(error_response.Code, final_res)
				return

			} else {
				c.JSON(200, res)
				return

			}
			break

		} else if duration > request_timeout {

			final_res := response.ErrorResponse{Success: false, Error: "failed to process request before timeout"}

			c.JSON(500, final_res)
			return
		}

		time.Sleep(5 * time.Millisecond)
	}
}

func detection(c *gin.Context, queue_name string) {

	nms := c.PostForm("min_confidence")

	if nms == "" {

		nms = "0.40"

	}

	img_id := uuid.NewV4().String()

	req_id := uuid.NewV4().String()

	detec_req := requests.DetectionRequest{Imgid: img_id, Minconfidence: nms, Reqtype: "detection", Reqid: req_id}

	face_req_string, _ := json.Marshal(detec_req)

	file, _ := c.FormFile("image")

	c.SaveUploadedFile(file, filepath.Join(temp_path, img_id))

	redis_client.RPush(queue_name, face_req_string)

	t1 := time.Now()

	for true {

		output, _ := redis_client.Get(req_id).Result()
		duration := time.Since(t1).Seconds()

		if output != "" {

			var res response.DetectionResponse

			json.Unmarshal([]byte(output), &res)

			if res.Success == false {

				var error_response response.ErrorResponseInternal
				json.Unmarshal([]byte(output), &error_response)

				final_res := response.ErrorResponse{Success: false, Error: error_response.Error}

				c.JSON(error_response.Code, final_res)
				return

			} else {
				c.JSON(200, res)

				return
			}

			break
		} else if duration > request_timeout {

			final_res := response.ErrorResponse{Success: false, Error: "failed to process request before timeout"}

			c.JSON(500, final_res)
			return
		}

		time.Sleep(1 * time.Millisecond)
	}
}

func facedetection(c *gin.Context) {

	file, _ := c.FormFile("image")

	nms := c.PostForm("min_confidence")

	if nms == "" {

		nms = "0.55"

	}

	img_id := uuid.NewV4().String()
	req_id := uuid.NewV4().String()

	face_req := requests.FaceDetectionRequest{Imgid: img_id, Reqtype: "detect", Reqid: req_id, Minconfidence: nms}

	face_req_string, _ := json.Marshal(face_req)

	c.SaveUploadedFile(file, filepath.Join(temp_path, img_id))

	redis_client.RPush("face_queue", face_req_string)

	t1 := time.Now()

	for true {

		output, _ := redis_client.Get(req_id).Result()
		duration := time.Since(t1).Seconds()

		if output != "" {

			var res response.FaceDetectionResponse
			json.Unmarshal([]byte(output), &res)

			if res.Success == false {

				var error_response response.ErrorResponseInternal
				json.Unmarshal([]byte(output), &error_response)

				final_res := response.ErrorResponse{Success: false, Error: error_response.Error}

				c.JSON(error_response.Code, final_res)

			} else {

				c.JSON(200, res)
				return
			}

			break
		} else if duration > request_timeout {

			final_res := response.ErrorResponse{Success: false, Error: "failed to process request before timeout"}

			c.JSON(500, final_res)
			return
		}

		time.Sleep(1 * time.Millisecond)
	}
}

func facerecognition(c *gin.Context) {

	file, _ := c.FormFile("image")

	threshold := c.PostForm("min_confidence")

	if threshold == "" {

		threshold = "0.67"

	}

	img_id := uuid.NewV4().String()
	req_id := uuid.NewV4().String()

	c.SaveUploadedFile(file, filepath.Join(temp_path, img_id))

	face_req := requests.FaceRecognitionRequest{Imgid: img_id, Reqtype: "recognize", Reqid: req_id, Minconfidence: threshold}

	face_req_string, _ := json.Marshal(face_req)

	redis_client.RPush("face_queue", face_req_string)

	t1 := time.Now()
	for true {

		output, _ := redis_client.Get(req_id).Result()
		duration := time.Since(t1).Seconds()

		if output != "" {

			var res response.FaceRecognitionResponse
			json.Unmarshal([]byte(output), &res)

			if res.Success == false {

				var error_response response.ErrorResponseInternal
				json.Unmarshal([]byte(output), &error_response)

				final_res := response.ErrorResponse{Success: false, Error: error_response.Error}
				c.JSON(error_response.Code, final_res)
				return

			} else {

				c.JSON(200, res)
				return
			}

			break
		} else if duration > request_timeout {

			final_res := response.ErrorResponse{Success: false, Error: "failed to process request before timeout"}

			c.JSON(500, final_res)
			return
		}

		time.Sleep(1 * time.Millisecond)
	}
}

func faceregister(c *gin.Context) {

	userid := c.PostForm("userid")

	form, _ := c.MultipartForm()

	user_images := []string{}

	if form != nil {
		for filename, _ := range form.File {
			file, _ := c.FormFile(filename)
			img_id := uuid.NewV4().String()
			c.SaveUploadedFile(file, filepath.Join(temp_path, img_id))

			user_images = append(user_images, img_id)
		}
	}

	req_id := uuid.NewV4().String()

	request_body := requests.FaceRegisterRequest{Userid: userid, Images: user_images, Reqid: req_id, Reqtype: "register"}

	request_string, _ := json.Marshal(request_body)

	redis_client.RPush("face_queue", request_string)

	t1 := time.Now()

	for true {

		output, _ := redis_client.Get(req_id).Result()
		duration := time.Since(t1).Seconds()

		if output != "" {

			var res response.FaceRegisterResponse
			json.Unmarshal([]byte(output), &res)

			if res.Success == false {

				var error_response response.ErrorResponseInternal
				json.Unmarshal([]byte(output), &error_response)

				final_res := response.ErrorResponse{Success: false, Error: error_response.Error}
				c.JSON(error_response.Code, final_res)
				return

			} else {
				c.JSON(200, res)
				return
			}

			break
		} else if duration > request_timeout {

			final_res := response.ErrorResponse{Success: false, Error: "failed to process request before timeout"}

			c.JSON(500, final_res)
			return
		}

		time.Sleep(1 * time.Millisecond)
	}
}

func facematch(c *gin.Context) {

	form, _ := c.MultipartForm()

	user_images := []string{}

	if form != nil {
		for filename, _ := range form.File {
			file, _ := c.FormFile(filename)
			img_id := uuid.NewV4().String()
			c.SaveUploadedFile(file, filepath.Join(temp_path, img_id))

			user_images = append(user_images, img_id)
		}
	}

	req_id := uuid.NewV4().String()

	request_body := requests.FaceMatchRequest{Images: user_images, Reqid: req_id, Reqtype: "match"}

	request_string, _ := json.Marshal(request_body)

	redis_client.RPush("face_queue", request_string)

	t1 := time.Now()

	for true {

		output, _ := redis_client.Get(req_id).Result()
		duration := time.Since(t1).Seconds()

		if output != "" {

			var res response.FaceMatchResponse
			json.Unmarshal([]byte(output), &res)

			if res.Success == false {

				var error_response response.ErrorResponseInternal
				json.Unmarshal([]byte(output), &error_response)

				final_res := response.ErrorResponse{Success: false, Error: error_response.Error}
				c.JSON(error_response.Code, final_res)
				return

			} else {
				c.JSON(200, res)
				return
			}

			break
		} else if duration > request_timeout {

			final_res := response.ErrorResponse{Success: false, Error: "failed to process request before timeout"}

			c.JSON(500, final_res)
			return
		}

		time.Sleep(1 * time.Millisecond)
	}

}

func listface(c *gin.Context) {

	TB_EMBEDDINGS := "TB_EMBEDDINGS"
	face2 := os.Getenv("VISION-FACE2")

	if face2 == "True" {

		TB_EMBEDDINGS = "TB_EMBEDDINGS2"

	}

	rows, _ := db.Query(fmt.Sprintf("select userid from %s", TB_EMBEDDINGS))

	var userids = []string{}
	for rows.Next() {

		var userid string
		rows.Scan(&userid)

		userids = append(userids, userid)

	}

	res := response.FacelistResponse{Success: true, Faces: userids}

	c.JSON(200, res)
	return

}

func deleteface(c *gin.Context) {

	userid := c.PostForm("userid")

	TB_EMBEDDINGS := "TB_EMBEDDINGS"
	face2 := os.Getenv("VISION-FACE2")

	if face2 == "True" {

		TB_EMBEDDINGS = "TB_EMBEDDINGS2"

	}

	trans, _ := db.Begin()

	stmt, _ := trans.Prepare(fmt.Sprintf("DELETE FROM %s WHERE userid=?", TB_EMBEDDINGS))

	defer stmt.Close()

	stmt.Exec(userid)

	trans.Commit()

	res := response.FaceDeleteResponse{Success: true}

	c.JSON(200, res)
	return

}

func register_model(c *gin.Context) {

	model_file, _ := c.FormFile("model")

	config_file, _ := c.FormFile("config")

	model_name := c.PostForm("name")

	MODEL_DIR := DATA_DIR + "/models/vision/" + model_name + "/"

	model_exists, _ := utils.PathExists(MODEL_DIR)
	message := "model updated"
	if model_exists == false {

		os.MkdirAll(MODEL_DIR, os.ModePerm)
		message = "model registered"

	}

	c.SaveUploadedFile(model_file, MODEL_DIR+"model.pb")
	c.SaveUploadedFile(config_file, MODEL_DIR+"config.json")
	res := response.ModelRegisterResponse{Success: true, Message: message}

	c.JSON(200, res)

}

func delete_model(c *gin.Context) {

	model_name := c.PostForm("name")

	MODEL_DIR := DATA_DIR + "/models/vision/" + model_name + "/"

	os.RemoveAll(MODEL_DIR)

	res := response.ModelDeleteResponse{Success: true, Message: "Model removed"}

	c.JSON(200, res)
	return

}

func list_models(c *gin.Context) {

	model_list, err := filepath.Glob(DATA_DIR + "/models/vision/*")

	models := []structures.ModelInfo{}

	if err == nil {

		for _, file := range model_list {

			model_name := filepath.Base(file)
			fileStat, _ := os.Stat(file + "/model.pb")
			size := float32(fileStat.Size()) / (1000 * 1000)

			model_info := structures.ModelInfo{Name: model_name, Dateupdated: fileStat.ModTime(), Modelsize: size}

			models = append(models, model_info)

		}

	}

	res := structures.AllModels{Models: models, Success: true}

	c.JSON(200, res)

	return

}

func single_request_loop(c *gin.Context, queue_name string) {

	img_id := uuid.NewV4().String()
	req_id := uuid.NewV4().String()

	file, _ := c.FormFile("image")

	c.SaveUploadedFile(file, filepath.Join(temp_path, img_id))

	req_data := requests.RecognitionRequest{Imgid: img_id, Reqid: req_id, Reqtype: "custom"}
	req_string, _ := json.Marshal(req_data)

	redis_client.RPush(queue_name, req_string)

	for true {

		output, _ := redis_client.Get(req_id).Result()

		if output != "" {

			var res response.RecognitionResponse
			json.Unmarshal([]byte(output), &res)

			if res.Success == false {

				var error_response response.ErrorResponseInternal

				json.Unmarshal([]byte(output), &error_response)

				final_res := response.ErrorResponse{Success: false, Error: error_response.Error}

				c.JSON(error_response.Code, final_res)
				return

			} else {
				c.JSON(200, res)
				return

			}

			break
		}

		time.Sleep(1 * time.Millisecond)
	}
}

func backup(c *gin.Context) {

	file_id := uuid.NewV4().String() + ".zip"
	backup_name := "Backup_" + time.Now().Format("2006-01-02T15:04:05") + ".backup"

	output_file, _ := os.Create(temp_path + "/" + file_id)

	zip_archive := zip.NewWriter(output_file)

	models, err := filepath.Glob(DATA_DIR + "/models/vision/*")

	if err == nil {

		for _, file := range models {

			model_name := filepath.Base(file)

			utils.AddFileToZip(zip_archive, path.Join(file, "model.pb"), "models/vision/"+model_name+"/model.pb")
			utils.AddFileToZip(zip_archive, path.Join(file, "config.json"), "models/vision/"+model_name+"/config.json")

		}

	}

	utils.AddFileToZip(zip_archive, DATA_DIR+"/faceembedding.db", "faceembedding.db")

	zip_archive.Close()
	output_file.Close()

	data_file, _ := os.Open(temp_path + "/" + file_id)

	info, err := os.Stat(temp_path + "/" + file_id)

	if err != nil {

		fmt.Println(err)
	}

	contentLength := info.Size()

	contentType := "application/octet-stream"

	extraHeaders := map[string]string{
		"Content-Disposition": "attachment; filename=" + backup_name,
	}

	c.DataFromReader(200, contentLength, contentType, data_file, extraHeaders)

}

func restore(c *gin.Context) {

	backup_file, _ := c.FormFile("file")

	backup_path := temp_path + "/deepstack.backup"
	c.SaveUploadedFile(backup_file, backup_path)
	defer os.Remove(backup_path)

	zip_reader, err := zip.OpenReader(backup_path)

	if err != nil {

		response := response.ErrorResponse{Success: false, Error: "Invalid backup file"}

		c.JSON(200, response)

		return
	}

	defer zip_reader.Close()

	for _, f := range zip_reader.File {

		f_path := f.Name
		data, err := f.Open()
		if err != nil {

			fmt.Println(err)
		}

		fpath := path.Join(DATA_DIR, f_path)

		os.MkdirAll(filepath.Dir(fpath), os.ModePerm)

		outFile, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())

		_, err = io.Copy(outFile, data)
		outFile.Close()

	}

	res := response.RestoreResponse{Success: true}

	c.JSON(200, res)

	return

}

func superresolution(c *gin.Context, queue_name string) {

	img_id := uuid.NewV4().String()

	req_id := uuid.NewV4().String()

	superres_req := requests.SuperresolutionRequest{Imgid: img_id, Reqtype: "superresolution", Reqid: req_id}

	superres_req_string, _ := json.Marshal(superres_req)

	file, _ := c.FormFile("image")

	c.SaveUploadedFile(file, filepath.Join(temp_path, img_id))

	redis_client.RPush(queue_name, superres_req_string)

	t1 := time.Now()

	for true {

		output, _ := redis_client.Get(req_id).Result()
		duration := time.Since(t1).Seconds()

		if output != "" {

			var res response.SuperresolutionResponse

			json.Unmarshal([]byte(output), &res)

			if res.Success == false {

				var error_response response.ErrorResponseInternal
				json.Unmarshal([]byte(output), &error_response)

				final_res := response.ErrorResponse{Success: false, Error: error_response.Error}

				c.JSON(error_response.Code, final_res)
				return

			} else {
				c.JSON(200, res)

				return
			}

			break
		} else if duration > request_timeout {

			final_res := response.ErrorResponse{Success: false, Error: "failed to process request before timeout"}

			c.JSON(500, final_res)
			return
		}

		time.Sleep(1 * time.Millisecond)
	}
}

func printfromprocess(cmd *exec.Cmd) {

	for true {

		out, err := cmd.StdoutPipe()
		if err == nil {

			outData, _ := ioutil.ReadAll(out)
			fmt.Println(string(outData))
			time.Sleep(1 * time.Second)

		}

	}

}

func printlogs() {

	face1 := os.Getenv("VISION-FACE")
	face2 := os.Getenv("VISION-FACE2")
	detection := os.Getenv("VISION-DETECTION")
	scene := os.Getenv("VISION-SCENE")
	enhance := os.Getenv("VISION-ENHANCE")

	if face1 == "True" || face2 == "True" {

		fmt.Println("/v1/vision/face")
		fmt.Println("---------------------------------------")
		fmt.Println("/v1/vision/face/recognize")
		fmt.Println("---------------------------------------")
		fmt.Println("/v1/vision/face/register")
		fmt.Println("---------------------------------------")
		fmt.Println("/v1/vision/face/match")
		fmt.Println("---------------------------------------")
		fmt.Println("/v1/vision/face/list")
		fmt.Println("---------------------------------------")
		fmt.Println("/v1/vision/face/delete")
		fmt.Println("---------------------------------------")

	}

	if detection == "True" {

		fmt.Println("/v1/vision/detection")
		fmt.Println("---------------------------------------")

	}

	if scene == "True" {

		fmt.Println("/v1/vision/scene")
		fmt.Println("---------------------------------------")

	}

	if enhance == "True" {

		fmt.Println("/v1/vision/enhance")
		fmt.Println("---------------------------------------")

	}

	models, err := filepath.Glob(DATA_DIR + "/models/vision/*")

	custom := os.Getenv("VISION-CUSTOM")

	if err == nil && custom == "True" {

		for _, file := range models {
			model_name := filepath.Base(file)
			fmt.Println("v1/vision/custom/" + model_name)
			fmt.Println("---------------------------------------")
		}

	}

	fmt.Println("---------------------------------------")
	fmt.Println("v1/backup")
	fmt.Println("---------------------------------------")
	fmt.Println("v1/restore")

}

func home(c *gin.Context) {

	c.HTML(200, "index.html", gin.H{})

}

func initActivation() {

	face := os.Getenv("VISION_FACE")
	detection := os.Getenv("VISION_DETECTION")
	scene := os.Getenv("VISION_SCENE")
	enhance := os.Getenv("VISION_ENHANCE")
	api_key := os.Getenv("API_KEY")
	send_logs := os.Getenv("SEND_LOGS")

	if os.Getenv("VISION-FACE") == "" {
		os.Setenv("VISION-FACE", face)
	}
	if os.Getenv("VISION-DETECTION") == "" {
		os.Setenv("VISION-DETECTION", detection)
	}
	if os.Getenv("VISION-SCENE") == "" {
		os.Setenv("VISION-SCENE", scene)
	}
	if os.Getenv("VISION-ENHANCE") == "" {
		os.Setenv("VISION-ENHANCE", enhance)
	}
	if os.Getenv("API-KEY") == "" {
		os.Setenv("API-KEY", api_key)
	}
	if os.Getenv("SEND-LOGS") == "" {
		os.Setenv("SEND-LOGS", send_logs)
	}
}

func launchservices() {

}

func main() {

	initActivation()

	var visionFace string
	var visionDetection string
	var visionScene string
	var visionEnhance string
	var apiKey string
	var adminKey string
	var port int
	var modelStoreDetection string
	var mode string
	var certPath string
	var sendLogs string

	if os.Getenv("PROFILE") == "" {
		os.Chdir("C://DeepStack//server")
		platformdata, err := ioutil.ReadFile("platform.json")

		if err == nil {
			var platform structures.PLATFORM

			json.Unmarshal(platformdata, &platform)

			os.Setenv("PROFILE", platform.PROFILE)
			os.Setenv("CUDA_MODE", platform.CUDA_MODE)
		}
	}

	versionfile, err := os.Open("version.txt")
	deepstack_version := ""

	if err == nil {
		versiondata, _ := ioutil.ReadAll(versionfile)
		deepstack_version = string(versiondata)

		fmt.Println("DeepStack: Version " + deepstack_version)
	}

	flag.StringVar(&visionFace, "VISION-FACE", os.Getenv("VISION-FACE"), "enable face detection")
	flag.StringVar(&visionDetection, "VISION-DETECTION", os.Getenv("VISION-DETECTION"), "enable object detection")
	flag.StringVar(&visionScene, "VISION-SCENE", os.Getenv("VISION-SCENE"), "enable scene recognition")
	flag.StringVar(&visionEnhance, "VISION-ENHANCE", os.Getenv("VISION-ENHANCE"), "enable image superresolution")
	flag.StringVar(&apiKey, "API-KEY", os.Getenv("API-KEY"), "api key to secure endpoints")
	flag.StringVar(&adminKey, "ADMIN-KEY", os.Getenv("ADMIN-KEY"), "admin key to secure admin endpoints")
	flag.StringVar(&modelStoreDetection, "MODELSTORE-DETECTION", "/modelstore/detection/", "path to custom detection models")
	flag.StringVar(&certPath, "CERT-PATH", "/cert", "path to ssl certificate files")

	floatTimeoutVal, err := strconv.ParseFloat(os.Getenv("TIMEOUT"), 32)
	if err != nil {
		flag.Float64Var(&request_timeout, "TIMEOUT", 60, "request timeout in seconds")
	} else {
		flag.Float64Var(&request_timeout, "TIMEOUT", floatTimeoutVal, "request timeout in seconds")
	}

	mode_val, mode_set := os.LookupEnv("MODE")

	if mode_set {
		flag.StringVar(&mode, "MODE", mode_val, "performance mode")
	} else {
		flag.StringVar(&mode, "MODE", "Medium", "performance mode")
	}

	send_logs_val, send_logs_set := os.LookupEnv("MODE")

	if send_logs_set {
		flag.StringVar(&sendLogs, "SEND-LOGS", send_logs_val, "log system info to deepquestai server")
	} else {
		flag.StringVar(&sendLogs, "SEND-LOGS", "True", "log system info to deepquestai server")
	}

	getPort := os.Getenv("PORT")
	intPortVal, err := strconv.Atoi(getPort)
	if err != nil {
		flag.IntVar(&port, "PORT", 5000, "port")
	} else {
		flag.IntVar(&port, "PORT", intPortVal, "port")
	}

	flag.Parse()

	PROFILE := os.Getenv("PROFILE")

	if !strings.HasSuffix(modelStoreDetection, "/") {
		modelStoreDetection = modelStoreDetection + "/"
	}

	if !strings.HasSuffix(certPath, "/") {
		certPath = certPath + "/"
	}

	APPDIR := os.Getenv("APPDIR")
	DATA_DIR = os.Getenv("DATA_DIR")

	startedProcesses := make([]*exec.Cmd, 0)

	redis_server := "redis-server"
	interpreter := "python3"

	if PROFILE == "windows_native" {

		APPDIR = "C://DeepStack"
		interpreter = filepath.Join(APPDIR, "interpreter", "python.exe")
		redis_server = filepath.Join(APPDIR, "redis", "redis-server.exe")

		os.Setenv("VISION-FACE", visionFace)
		os.Setenv("VISION-DETECTION", visionDetection)
		os.Setenv("VISION-SCENE", visionScene)
		os.Setenv("VISION-ENHANCE", visionEnhance)
		os.Setenv("APPDIR", APPDIR)
		os.Setenv("MODE", mode)
	}

	if DATA_DIR == "" {
		DATA_DIR = "/datastore"

		if PROFILE == "windows_native" {
			DATA_DIR = filepath.Join(os.Getenv("LocalAppData"), "DeepStack")
		}
	}

	temp_path = os.Getenv("TEMP_PATH")
	if temp_path == "" {
		temp_path = "/deeptemp/"

		if PROFILE == "windows_native" {
			temp_path = filepath.Join(os.TempDir(), "DeepStack")
		}
	}

	logdir := filepath.Join(APPDIR, "logs")

	if PROFILE == "windows_native" {
		os.Setenv("DATA_DIR", DATA_DIR)
		os.Setenv("TEMP_PATH", temp_path)
		logdir = filepath.Join(DATA_DIR, "logs")
	}

	request_timeout_str := os.Getenv("TIMEOUT")
	request_timeout_val, err := strconv.ParseFloat(request_timeout_str, 64)

	if request_timeout_str != "" && err == nil {
		request_timeout = request_timeout_val
	}

	os.Mkdir(logdir, 0755)
	os.Mkdir(DATA_DIR, 0755)
	os.Mkdir(temp_path, 0755)

	if PROFILE == "windows_native" {
		go utils.CreateDirs(logdir, DATA_DIR, temp_path)
	}

	stdout, _ := os.Create(filepath.Join(logdir, "stdout.txt"))

	defer stdout.Close()

	stderr, _ := os.Create(filepath.Join(logdir, "stderr.txt"))

	defer stderr.Close()

	ctx := context.TODO()

	initScript := filepath.Join(APPDIR, "init.py")
	detectionScript := filepath.Join(APPDIR, "intelligencelayer/shared/detection.py")
	faceScript := filepath.Join(APPDIR, "intelligencelayer/shared/face.py")
	sceneScript := filepath.Join(APPDIR, "intelligencelayer/shared/scene.py")
	enhanceScript := filepath.Join(APPDIR, "intelligencelayer/shared/superresolution.py")

	initcmd := exec.CommandContext(ctx, "bash", "-c", interpreter+" "+initScript)
	if PROFILE == "windows_native" {
		initcmd = exec.CommandContext(ctx, interpreter, initScript)
	}
	initcmd.Dir = APPDIR
	initcmd.Stdout = stdout
	initcmd.Stderr = stderr

	rediscmd := exec.CommandContext(ctx, "bash", "-c", redis_server+" --daemonize yes")
	if PROFILE == "windows_native" {
		rediscmd = exec.CommandContext(ctx, redis_server)
		rediscmd.Dir = filepath.Join(APPDIR, "redis")
	}

	rediscmd.Stdout = stdout
	rediscmd.Stderr = stderr

	err = rediscmd.Start()
	if err != nil {
		stderr.WriteString("Redis server failed to start: " + err.Error())
	}
	err = initcmd.Run()
	startedProcesses = append(startedProcesses, initcmd)
	if err != nil {
		stderr.WriteString("Init process failed to start " + err.Error())
	}

	redis_client = redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "",
		DB:       0,
	})

	if visionDetection == "True" {
		detectioncmd := exec.CommandContext(ctx, "bash", "-c", interpreter+" "+detectionScript)
		if PROFILE == "windows_native" {
			detectioncmd = exec.CommandContext(ctx, interpreter, detectionScript)
		}
		startedProcesses = append(startedProcesses, detectioncmd)
		detectioncmd.Dir = filepath.Join(APPDIR, "intelligencelayer/shared")
		detectioncmd.Stdout = stdout
		detectioncmd.Stderr = stderr
		detectioncmd.Env = os.Environ()

		err = detectioncmd.Start()
		if err != nil {
			stderr.WriteString("Detection process failed to start" + err.Error())
		}

		// go utils.KeepProcess(detectioncmd, redis_client, "detection", PROFILE, interpreter, detectionScript, APPDIR, stdout, stderr, &ctx, startedProcesses)

	}

	if visionFace == "True" {
		facecmd := exec.CommandContext(ctx, "bash", "-c", interpreter+" "+faceScript)
		if PROFILE == "windows_native" {
			facecmd = exec.CommandContext(ctx, interpreter, faceScript)
		}
		startedProcesses = append(startedProcesses, facecmd)
		facecmd.Dir = filepath.Join(APPDIR, "intelligencelayer/shared")
		facecmd.Stdout = stdout
		facecmd.Stderr = stderr
		facecmd.Env = os.Environ()
		err = facecmd.Start()
		if err != nil {
			stderr.WriteString("face process failed to start " + err.Error())
		}
		// go utils.KeepProcess(facecmd, redis_client, "face", PROFILE, interpreter, faceScript, APPDIR, stdout, stderr, &ctx, startedProcesses)

	}
	if visionScene == "True" {
		scenecmd := exec.CommandContext(ctx, "bash", "-c", interpreter+" "+sceneScript)
		if PROFILE == "windows_native" {
			scenecmd = exec.CommandContext(ctx, interpreter, sceneScript)
		}

		startedProcesses = append(startedProcesses, scenecmd)
		scenecmd.Dir = filepath.Join(APPDIR, "intelligencelayer/shared")
		scenecmd.Stdout = stdout
		scenecmd.Stderr = stderr
		scenecmd.Env = os.Environ()
		err = scenecmd.Start()
		if err != nil {
			stderr.WriteString("scene process failed to start: " + err.Error())
		}
		// go utils.KeepProcess(scenecmd, redis_client, "scene", PROFILE, interpreter, sceneScript, APPDIR, stdout, stderr, &ctx, startedProcesses)

	}

	if visionEnhance == "True" {
		enhancecmd := exec.CommandContext(ctx, "bash", "-c", interpreter+" "+enhanceScript)
		if PROFILE == "windows_native" {
			enhancecmd = exec.CommandContext(ctx, interpreter, enhanceScript)
		}

		startedProcesses = append(startedProcesses, enhancecmd)
		enhancecmd.Dir = filepath.Join(APPDIR, "intelligencelayer/shared")
		enhancecmd.Stdout = stdout
		enhancecmd.Stderr = stderr
		enhancecmd.Env = os.Environ()
		err = enhancecmd.Start()
		if err != nil {
			stderr.WriteString("scene process failed to start: " + err.Error())
		}
		// go utils.KeepProcess(scenecmd, redis_client, "scene", PROFILE, interpreter, enhanceScript, APPDIR, stdout, stderr, &ctx, startedProcesses)

	}

	db, _ = sql.Open("sqlite3", filepath.Join(DATA_DIR, "faceembedding.db"))

	gin.SetMode(gin.ReleaseMode)

	server := gin.New()

	admin_key := os.Getenv("ADMIN-KEY")
	api_key := os.Getenv("API-KEY")

	if admin_key != "" || api_key != "" {

		if admin_key != "" {

			settings.ADMIN_KEY = admin_key

		} else {

			settings.ADMIN_KEY = ""

		}

		if api_key != "" {

			settings.API_KEY = api_key

		} else {

			settings.API_KEY = ""

		}

	}

	num_custom_det_models := 0

	server.Use(gin.Recovery())

	v1 := server.Group("/v1")
	v1.Use(gin.Logger())

	vision := v1.Group("/vision")
	vision.Use(middlewares.CheckApiKey(&sub_data, &settings))
	{
		vision.POST("/scene", middlewares.CheckScene(), middlewares.CheckImage(), scene)
		vision.POST("/detection", middlewares.CheckDetection(), middlewares.CheckImage(), middlewares.CheckConfidence(), func(c *gin.Context) {

			detection(c, "detection_queue")

		})
		vision.POST("/enhance", middlewares.CheckSuperresolution(), middlewares.CheckImage(), func(c *gin.Context) {

			superresolution(c, "superresolution_queue")

		})

		facegroup := vision.Group("/face")
		facegroup.Use(middlewares.CheckFace())
		{
			facegroup.POST("/", middlewares.CheckImage(), middlewares.CheckConfidence(), facedetection)
			facegroup.POST("/recognize", middlewares.CheckImage(), middlewares.CheckConfidence(), facerecognition)
			facegroup.POST("/register", middlewares.CheckMultiImage(), middlewares.CheckUserID(), faceregister)
			facegroup.POST("/match", middlewares.CheckFaceMatch(), facematch)
			facegroup.POST("/delete", middlewares.CheckUserID(), deleteface)
			facegroup.POST("/list", listface)

		}

		vision.POST("/addmodel", middlewares.CheckAdminKey(&sub_data, &settings), middlewares.CheckRegisterModel(&sub_data, DATA_DIR), register_model)
		vision.POST("/deletemodel", middlewares.CheckAdminKey(&sub_data, &settings), middlewares.CheckDeleteModel(DATA_DIR), delete_model)
		vision.POST("/listmodels", middlewares.CheckAdminKey(&sub_data, &settings), list_models)

		custom := vision.Group("/custom")
		custom.Use(middlewares.CheckImage())
		{

			models, err := filepath.Glob(modelStoreDetection + "*.pt")

			num_custom_det_models = len(models)

			if err == nil {

				for _, file := range models {

					model_name := filepath.Base(file)

					model_name = model_name[:strings.LastIndex(model_name, ".")]

					modelcmd := exec.CommandContext(ctx, "bash", "-c", interpreter+" "+detectionScript+" --model "+file+" --name "+model_name)
					if PROFILE == "windows_native" {
						modelcmd = exec.CommandContext(ctx, interpreter, detectionScript, "--model", file, "--name", model_name)
					}
					startedProcesses = append(startedProcesses, modelcmd)
					modelcmd.Dir = filepath.Join(APPDIR, "intelligencelayer/shared")
					modelcmd.Stdout = stdout
					modelcmd.Stderr = stderr
					err = modelcmd.Start()
					if err != nil {
						stderr.WriteString(err.Error())
					}

					custom.POST(model_name, func(c *gin.Context) {

						detection(c, model_name+"_queue")

					})

					fmt.Println("---------------------------------------")
					fmt.Println("v1/vision/custom/" + model_name)

				}

			}
		}

	}

	v1.POST("/backup", middlewares.CheckAdminKey(&sub_data, &settings), backup)
	v1.POST("/restore", middlewares.CheckAdminKey(&sub_data, &settings), middlewares.CheckRestore(), restore)

	server.Static("/assets", "./assets")
	server.LoadHTMLGlob("templates/*")
	server.GET("/", home)
	server.GET("/admin", home)

	if sendLogs == "True" {
		go utils.LogToServer(&sub_data, PROFILE, deepstack_version, (visionEnhance == "True"), (visionDetection == "True"), (visionFace == "True"), (visionScene == "True"), num_custom_det_models)
	}
	port2 := strconv.Itoa(port)
	printlogs()

	signalChannel := make(chan os.Signal, 2)
	signal.Notify(signalChannel, os.Interrupt, syscall.SIGTERM)
	go func() {
		sig := <-signalChannel
		if sig == syscall.SIGTERM || sig == syscall.SIGKILL {
			for _, process := range startedProcesses {
				err = process.Process.Kill()
				if err != nil {
					stderr.WriteString(err.Error())
				}
			}
		}
	}()

	fullChain := filepath.Join(certPath, "fullchain.pem")
	key := filepath.Join(certPath, "key.pem")

	fullchain_exists, _ := utils.PathExists(fullChain)
	key_exists, _ := utils.PathExists(key)

	if fullchain_exists == true && key_exists == true {
		fmt.Println("DeepStack is Running on HTTPS")
		server.RunTLS(":443", fullChain, key)
	} else {
		server.Run(":" + port2)
	}

}
