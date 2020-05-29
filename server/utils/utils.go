package utils

import (
	"archive/zip"
	"io"
	"os"
	"time"

	"deepstack.io/server/structures"
	"github.com/imroc/req"
	"github.com/shirou/gopsutil/cpu"
	"github.com/shirou/gopsutil/host"
)

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
