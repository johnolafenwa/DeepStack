using System;
using CommandLine;
using System.Threading;
using System.Diagnostics;
using System.IO;

namespace dotnet_runner
{
    public class CommandOptions
    {
        [Option("VISION-FACE", Required = false, Default = "False", HelpText = "enable face detection")]
        public string VisionFace { get; set; }

        [Option("VISION-DETECTION", Required = false, Default = "False", HelpText = "enable object detection")]
        public string VisionDetection { get; set; }

        [Option("VISION-SCENE", Required = false, Default ="False", HelpText = "enable scene recognition")]
        public string VisionScene { get; set; }

        [Option("PORT", Required = false, Default = 5000, HelpText = "specify port")]
        public int PORT { get; set; }

        [Option("API-KEY", Required = false, HelpText = "Specifies key to use for endpoints")]
        public string ApiKey { get; set; }

        [Option("ADMIN-KEY", Required = false, HelpText = "Specifies key to use for admin operations")]
        public string AdminKey { get; set; }

        [Option("MODELSTORE-DETECTION", Required = false, HelpText = "Patht to custom detection models")]
        public string ModelStoreDetection { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Starting DeepStack\n");

            Parser.Default.ParseArguments<CommandOptions>(args)
                   .WithParsed<CommandOptions>(o =>
                   {

                       string data_dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "DeepStack");
                       string temp_dir = Path.Combine(Path.GetTempPath(), "DeepStack");

                       if (!File.Exists(data_dir))
                       {
                           Directory.CreateDirectory(data_dir);
                       }

                       if (!File.Exists(temp_dir))
                       {
                           Directory.CreateDirectory(temp_dir);
                       }

                       Process process = new Process();
                       process.StartInfo.EnvironmentVariables.Add("VISION-FACE", o.VisionFace);
                       process.StartInfo.EnvironmentVariables.Add("VISION-DETECTION", o.VisionDetection);
                       process.StartInfo.EnvironmentVariables.Add("VISION-SCENE", o.VisionScene);
                       process.StartInfo.EnvironmentVariables.Add("API-KEY", o.ApiKey);
                       process.StartInfo.EnvironmentVariables.Add("ADMIN-KEY", o.AdminKey);
                       process.StartInfo.EnvironmentVariables.Add("PROFILE", "windows_native");
                       process.StartInfo.EnvironmentVariables.Add("APPDIR", @"C:\Users\johnolafenwa\Documents\AI\DeepStack");
                       process.StartInfo.EnvironmentVariables.Add("DATA_DIR", data_dir);
                       process.StartInfo.EnvironmentVariables.Add("TEMP_PATH", temp_dir);
                       process.StartInfo.EnvironmentVariables.Add("PORT", o.PORT.ToString());
                       process.StartInfo.EnvironmentVariables.Add("MODELSTORE-DETECTION", o.ModelStoreDetection);


                       process.StartInfo.FileName = @"C:\Users\johnolafenwa\Documents\AI\DeepStack\server\server.exe";
                       process.StartInfo.WorkingDirectory = @"C:\Users\johnolafenwa\Documents\AI\DeepStack\server"; //process.StartInfo.Arguments = String.Format("-PORT={0}",port);
                       process.StartInfo.CreateNoWindow = false;
                       process.StartInfo.UseShellExecute = false;
                       //process.StartInfo.RedirectStandardOutput = true;
                       //process.OutputDataReceived += new DataReceivedEventHandler(handlelogs);
                       process.Start();
                       
                 
                       //process.BeginOutputReadLine();
                   });
        }

        static void handlelogs(object sender, DataReceivedEventArgs line)
        {

            Console.WriteLine(line.Data);
         
        }

    }
}
