using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnrealBuildTool;

// Rubble's C ABI (engine/crates/engine-ffi), linked as a static library.
//
// The library is built by scripts/unreal/build-ffi.sh, which this module runs on every build (cargo
// no-ops in well under a second when nothing changed). Set RUBBLE_FFI_PREBUILT=1 to skip that and use
// whatever build-ffi.sh last produced. The header comes straight from engine/crates/engine-ffi/include, and
// the build fails if the library was built from a different header (rubble_ffi.stamp).
public class RubbleFFI : ModuleRules
{
	public RubbleFFI(ReadOnlyTargetRules Target) : base(Target)
	{
		Type = ModuleType.External;

		string RepoRoot = Path.GetFullPath(Path.Combine(ModuleDirectory, "..", "..", "..", "..", "..", "..", ".."));
		string HeaderDir = Path.Combine(RepoRoot, "engine", "crates", "engine-ffi", "include");
		string Header = Path.Combine(HeaderDir, "rubble.h");
		if (!File.Exists(Header))
		{
			throw new BuildException("RubbleFFI: header not found at " + Header + " (is the plugin inside the rubble repo?)");
		}
		PublicIncludePaths.Add(HeaderDir);
		// rubble_ffi_build.h (the library's hash; see build-ffi.sh) forces a relink on change.
		PublicIncludePaths.Add(Path.Combine(ModuleDirectory, "include-gen"));

		string Platform;
		string Arch;
		string LibName;
		if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			Platform = "Mac"; Arch = "arm64"; LibName = "librubble_ffi.a";
		}
		else if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			Platform = "Linux"; Arch = "x86_64"; LibName = "librubble_ffi.a";
		}
		else if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			Platform = "Win64"; Arch = "x64"; LibName = "rubble_ffi.lib";
		}
		else
		{
			throw new BuildException("RubbleFFI: unsupported platform " + Target.Platform);
		}

		if (Environment.GetEnvironmentVariable("RUBBLE_FFI_PREBUILT") != "1")
		{
			RunBuildCore(RepoRoot, Platform);
		}

		string LibDir = Path.Combine(ModuleDirectory, "lib", Platform, Arch);
		string LibPath = Path.Combine(LibDir, LibName);
		if (!File.Exists(LibPath))
		{
			throw new BuildException("RubbleFFI: " + LibPath + " missing; run scripts/unreal/build-ffi.sh");
		}
		CheckStamp(Path.Combine(LibDir, "rubble_ffi.stamp"), Header);

		PublicAdditionalLibraries.Add(LibPath);
		AddNativeLibs(Path.Combine(LibDir, "native-libs.txt"));
	}

	private static void RunBuildCore(string RepoRoot, string Platform)
	{
		string Script = Path.Combine(RepoRoot, "scripts", "unreal", "build-ffi.sh");
		ProcessStartInfo Info = new ProcessStartInfo("/bin/bash", "\"" + Script + "\" --platform " + Platform)
		{
			UseShellExecute = false,
			RedirectStandardOutput = true,
			RedirectStandardError = true,
			WorkingDirectory = RepoRoot,
		};
		using (Process Build = Process.Start(Info))
		{
			string Out = Build.StandardOutput.ReadToEnd();
			string Err = Build.StandardError.ReadToEnd();
			Build.WaitForExit();
			if (Build.ExitCode != 0)
			{
				throw new BuildException("RubbleFFI: build-ffi.sh failed (" + Build.ExitCode + ")\n" + Out + Err);
			}
			string Summary = Out.Split('\n').LastOrDefault(Line => Line.StartsWith("build-ffi:"));
			Console.WriteLine(Summary ?? "RubbleFFI: built");
		}
	}

	private static void CheckStamp(string StampPath, string Header)
	{
		if (!File.Exists(StampPath))
		{
			throw new BuildException("RubbleFFI: " + StampPath + " missing; run scripts/unreal/build-ffi.sh");
		}
		string Expected = File.ReadAllLines(StampPath)
			.Where(Line => Line.StartsWith("header_sha256="))
			.Select(Line => Line.Substring("header_sha256=".Length).Trim())
			.FirstOrDefault() ?? "";
		string Actual;
		using (SHA256 Sha = SHA256.Create())
		{
			Actual = BitConverter.ToString(Sha.ComputeHash(File.ReadAllBytes(Header))).Replace("-", "").ToLowerInvariant();
		}
		if (Expected != Actual)
		{
			throw new BuildException("RubbleFFI: library was built from a different header (stamp " + Expected + ", header " + Actual + "); run scripts/unreal/build-ffi.sh");
		}
	}

	private void AddNativeLibs(string NativeLibsPath)
	{
		if (!File.Exists(NativeLibsPath))
		{
			return;
		}
		string[] Tokens = File.ReadAllText(NativeLibsPath).Split(new[] { ' ', '\n', '\t' }, StringSplitOptions.RemoveEmptyEntries);
		for (int i = 0; i < Tokens.Length; i++)
		{
			string Token = Tokens[i];
			if (Token == "-framework" && i + 1 < Tokens.Length)
			{
				PublicFrameworks.Add(Tokens[++i]);
			}
			else if (Token.StartsWith("-l"))
			{
				string Name = Token.Substring(2);
				// libSystem (and its libc/libm aliases) is always linked on Apple platforms.
				if (Name != "System" && Name != "c" && Name != "m")
				{
					PublicSystemLibraries.Add(Name);
				}
			}
			else if (Token.EndsWith(".lib"))
			{
				PublicSystemLibraries.Add(Token);
			}
		}
	}
}
