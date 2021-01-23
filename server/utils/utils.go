package utils

import (
	"archive/zip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"

	"deepstack.io/server/requests"
	"deepstack.io/server/response"
	"deepstack.io/server/structures"
	"github.com/go-redis/redis"
	"github.com/imroc/req"
	"github.com/shirou/gopsutil/cpu"
	"github.com/shirou/gopsutil/host"
)

func ProcessExists(pid int32) (bool, error) {
	if pid <= 0 {
		return false, fmt.Errorf("invalid pid %v", pid)
	}
	proc, err := os.FindProcess(int(pid))
	if err != nil {
		return false, err
	}
	err = proc.Signal(syscall.Signal(0))
	if err == nil {
		return true, nil
	}
	if err.Error() == "os: process already finished" {
		return false, nil
	}
	errno, ok := err.(syscall.Errno)
	if !ok {
		return false, err
	}
	switch errno {
	case syscall.ESRCH:
		return false, nil
	case syscall.EPERM:
		return true, nil
	}
	return false, err
}

func CreateDirs(dirList ...string) {
	for true {
		for _, dir := range dirList {
			pathExists, _ := PathExists(dir)
			if pathExists == false {
				os.Mkdir(dir, 0755)
			}
		}
		time.Sleep(1 * time.Second)
	}
}

func PathExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return true, err
}

func AddFileToZip(zipWriter *zip.Writer, filename string, outname string) error {

	fileToZip, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer fileToZip.Close()

	// Get the file information
	info, err := fileToZip.Stat()
	if err != nil {
		return err
	}

	header, err := zip.FileInfoHeader(info)
	if err != nil {
		return err
	}

	// Using FileInfoHeader() above only uses the basename of the file. If we want
	// to preserve the folder structure we can overwrite this with the full path.
	header.Name = outname

	// Change to deflate to gain better compression
	// see http://golang.org/pkg/archive/zip/#pkg-constants
	header.Method = zip.Deflate

	writer, err := zipWriter.CreateHeader(header)
	if err != nil {
		return err
	}
	_, err = io.Copy(writer, fileToZip)
	return err
}

func AddRegisteryEntry(value string) {

}

func ReadRegisteryEntry() string {

	return ""

}

func LogToServer(sub_data *structures.ActivationData) {

	for true {

		h, _ := host.Info()
		c, _ := cpu.Info()

		platform := h.Platform
		os := h.OS
		osversion := h.PlatformVersion
		num_cores := len(c)
		cpu := ""
		for _, v := range c {
			cpu = v.ModelName
			break
		}

		params := req.Param{
			"key":       sub_data.Key,
			"cores":     num_cores,
			"cpu":       cpu,
			"platform":  platform,
			"os":        os,
			"osversion": osversion,
			"time":      time.Now(),
		}

		req.Post("https://register.deepstack.cc/loguser", params)
		time.Sleep(86400 * time.Second)

	}

}

// extend to support custom models
func KeepProcess(proc *exec.Cmd, redis_client *redis.Client, key string, profile string, interpreter string, script string, APPDIR string, stdout *os.File, stderr *os.File, ctx *context.Context, startedProcesses []*exec.Cmd) {
	for true {

		processExists, _ := ProcessExists(int32(proc.Process.Pid))
		if processExists == false {
			allrequests := redis_client.LRange(key, 0, 0)
			if allrequests.Err() == nil {

				singlerequest, err := allrequests.Result()
				if err == nil {
					for _, datastr := range singlerequest {

						if key == "scene" {
							var data requests.Request
							json.Unmarshal([]byte(datastr), &data)

							var res response.ErrorResponseInternal
							res.Code = 500
							res.Error = "process exited within deepstack, internal restart in process"
							res.Success = false

							res_string, _ := json.Marshal(res)
							redis_client.Set(data.Reqid, res_string, 1*time.Hour)
						}

					}
				}
			}

			procmd := exec.CommandContext(*ctx, "bash", "-c", interpreter+" "+script)
			if profile == "windows_native" {
				procmd = exec.CommandContext(*ctx, interpreter, script)
			}

			startedProcesses = append(startedProcesses, procmd)
			procmd.Dir = filepath.Join(APPDIR, "intelligencelayer/shared")
			procmd.Stdout = stdout
			procmd.Stderr = stderr
			procmd.Env = os.Environ()
			err := procmd.Start()
			if err != nil {
				stderr.WriteString(key + " process failed to re-start: will retry " + err.Error())
			} else {
				// go KeepProcess(procmd, redis_client, key, profile, interpreter, script, APPDIR, stdout, stderr, ctx, startedProcesses)
				break
			}

		}
		time.Sleep(5 * time.Second)
	}

}
