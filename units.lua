require 'tundra.syntax.glob'
require 'tundra.syntax.files'

local winFilter = "win*"
local winDebugFilter = "win*-*-debug"
local winReleaseFilter = "win*-*-release"
local linuxFilter = "linux*"

local winKernelLibs = { "kernel32.lib", "user32.lib", "gdi32.lib", "winspool.lib", "advapi32.lib", "shell32.lib", "comctl32.lib", 
						"uuid.lib", "ole32.lib", "oleaut32.lib", "shlwapi.lib", "OLDNAMES.lib", "wldap32.lib", "wsock32.lib",
						"Psapi.lib", "Msimg32.lib", "Comdlg32.lib", "RpcRT4.lib", "Iphlpapi.lib", "Delayimp.lib" }

-- Dynamic (msvcr110.dll etc) CRT linkage
local winLibsDynamicCRTDebug = tundra.util.merge_arrays( { "msvcrtd.lib", "msvcprtd.lib", "comsuppwd.lib" }, winKernelLibs )
local winLibsDynamicCRTRelease = tundra.util.merge_arrays( { "msvcrt.lib", "msvcprt.lib", "comsuppw.lib" }, winKernelLibs )

winLibsDynamicCRTDebug.Config = winDebugFilter
winLibsDynamicCRTRelease.Config = winReleaseFilter

-- Static CRT linkage
local winLibsStaticCRTDebug = tundra.util.merge_arrays( { "libcmtd.lib", "libcpmtd.lib", "comsuppwd.lib" }, winKernelLibs )
local winLibsStaticCRTRelease = tundra.util.merge_arrays( { "libcmt.lib", "libcpmt.lib", "comsuppw.lib" }, winKernelLibs )

winLibsStaticCRTDebug.Config = winDebugFilter
winLibsStaticCRTRelease.Config = winReleaseFilter

local winDynamicOpts = {
	{ "/MDd";					Config = winDebugFilter },
	{ "/MD";					Config = winReleaseFilter },
}

local winStaticOpts = {
	{ "/MTd";					Config = winDebugFilter },
	{ "/MT";					Config = winReleaseFilter },
}

local winDefs = {
	{ "_CRTDBG_MAP_ALLOC";		Config = winDebugFilter },
}

local winDynamicEnv = {
	CCOPTS = winDynamicOpts,
	CXXOPTS = winDynamicOpts,
	CCDEFS = winDefs,
	CPPDEFS = winDefs,
}

local winStaticEnv = {
	CCOPTS = winStaticOpts,
	CXXOPTS = winStaticOpts,
	CCDEFS = winDefs,
	CPPDEFS = winDefs,
}

local crtDynamic = ExternalLibrary {
	Name = "crtdynamic",
	Propagate = {
		Env = winDynamicEnv,
		Libs = {
			winLibsDynamicCRTDebug,
			winLibsDynamicCRTRelease,
		},
	},
}

local crtStatic = ExternalLibrary {
	Name = "crtstatic",
	Propagate = {
		Env = winStaticEnv,
		Libs = {
			winLibsStaticCRTDebug,
			winLibsStaticCRTRelease,
		},
	},
}

-- Swapping this out will change linkage to use MSVCR120.dll and its cousins,
-- instead of statically linking to the required MSVC runtime libraries.
-- I don't understand why, but if we build with crtStatic, then tests fail
-- due to heap corruption. I can only guess it's the old issue of two different
-- libraries using different heap properties. For example, library A allocated
-- memory with debug_alloc(), and then library B tries to free that memory with
-- regular free(). However, I cannot figure out how that is happening.
--local winCrt = crtStatic
local winCrt = crtDynamic

local linuxCrt = ExternalLibrary {
	Name = "linuxCrt",
	Propagate = {
		Libs = {
			{ "stdc++"; Config = linuxFilter }
		},
	},
}

-- Return an FGlob node that has our standard filters applied
local function makeGlob(dir, options)
	local filters = {
		{ Pattern = "_windows"; Config = winFilter },
		{ Pattern = "_linux"; Config = linuxFilter },
		{ Pattern = "_android"; Config = "ignore" },       -- Android stuff is built with a different build system
		{ Pattern = "[/\\]_[^/\\]*$"; Config = "ignore" }, -- Any file that starts with an underscore is ignored
	}
	if options.Ignore ~= nil then
		for _, ignore in ipairs(options.Ignore) do
			filters[#filters + 1] = { Pattern = ignore; Config = "ignore" }
		end
	end

	return FGlob {
		Dir = dir,
		Extensions = { ".c", ".cpp", ".h" },
		Filters = filters,
	}
end

local function copyfile_to_output(source, config)
	-- extract just the final part of the path (ie the filename)
	local filename = source:match("/([^/$]+)$")

	if config then
		return CopyFile { Source = source, Target = "$(OBJECTDIR)$(SEP)" .. filename; Config = config }
	else
		return CopyFile { Source = source, Target = "$(OBJECTDIR)$(SEP)" .. filename }
	end
end

local ideHintThirdParty = {
	Msvc = {
		SolutionFolder = "Third Party"
	}
}

local ideHintApp = {
	Msvc = {
		SolutionFolder = "Applications"
	}
}

local unicode = ExternalLibrary {
	Name = "unicode",
	Propagate = {
		Defines = { "UNICODE", "_UNICODE" },
	},
}

local vcpkg_bin = "third_party/vcpkg/installed/x64-windows/"

local deploy_libcurl_debug = copyfile_to_output(vcpkg_bin .. "debug/bin/libcurl.dll", winDebugFilter)
local deploy_libcurl_release = copyfile_to_output(vcpkg_bin .. "bin/libcurl.dll", winReleaseFilter)

local deploy_libssh2_debug = copyfile_to_output(vcpkg_bin .. "debug/bin/libssh2.dll", winDebugFilter)
local deploy_libssh2_release = copyfile_to_output(vcpkg_bin .. "bin/libssh2.dll", winReleaseFilter)

local deploy_libeay32_debug = copyfile_to_output(vcpkg_bin .. "debug/bin/libeay32.dll", winDebugFilter)
local deploy_libeay32_release = copyfile_to_output(vcpkg_bin .. "bin/libeay32.dll", winReleaseFilter)

local deploy_ssleay32_debug = copyfile_to_output(vcpkg_bin .. "debug/bin/ssleay32.dll", winDebugFilter)
local deploy_ssleay32_release = copyfile_to_output(vcpkg_bin .. "bin/ssleay32.dll", winReleaseFilter)

local deploy_zlib_debug = copyfile_to_output(vcpkg_bin .. "debug/bin/zlibd1.dll", winDebugFilter)
local deploy_zlib_release = copyfile_to_output(vcpkg_bin .. "bin/zlib1.dll", winReleaseFilter)

local deploy_lz4 = copyfile_to_output(vcpkg_bin .. "bin/lz4.dll", winFilter)

local deploy_avcodec_debug = copyfile_to_output(vcpkg_bin .. "debug/bin/avcodec-57.dll", winDebugFilter)
local deploy_avdevice_debug = copyfile_to_output(vcpkg_bin .. "debug/bin/avdevice-57.dll", winDebugFilter)
local deploy_avfilter_debug = copyfile_to_output(vcpkg_bin .. "debug/bin/avfilter-6.dll", winDebugFilter)
local deploy_avformat_debug = copyfile_to_output(vcpkg_bin .. "debug/bin/avformat-57.dll", winDebugFilter)
local deploy_avutil_debug = copyfile_to_output(vcpkg_bin .. "debug/bin/avutil-55.dll", winDebugFilter)
local deploy_swresample_debug = copyfile_to_output(vcpkg_bin .. "debug/bin/swresample-2.dll", winDebugFilter)
local deploy_swscale_debug = copyfile_to_output(vcpkg_bin .. "debug/bin/swscale-4.dll", winDebugFilter)

local deploy_avcodec_release = copyfile_to_output(vcpkg_bin .. "bin/avcodec-57.dll", winReleaseFilter)
local deploy_avdevice_release = copyfile_to_output(vcpkg_bin .. "bin/avdevice-57.dll", winReleaseFilter)
local deploy_avfilter_release = copyfile_to_output(vcpkg_bin .. "bin/avfilter-6.dll", winReleaseFilter)
local deploy_avformat_release = copyfile_to_output(vcpkg_bin .. "bin/avformat-57.dll", winReleaseFilter)
local deploy_avutil_release = copyfile_to_output(vcpkg_bin .. "bin/avutil-55.dll", winReleaseFilter)
local deploy_swresample_release = copyfile_to_output(vcpkg_bin .. "bin/swresample-2.dll", winReleaseFilter)
local deploy_swscale_release = copyfile_to_output(vcpkg_bin .. "bin/swscale-4.dll", winReleaseFilter)

local deploy_xo_debug = copyfile_to_output("third_party/xo/t2-output/win64-msvc2015-debug-default/xo.dll", winDebugFilter)
local deploy_xo_release = copyfile_to_output("third_party/xo/t2-output/win64-msvc2015-release-default/xo.dll", winReleaseFilter)

local ffmpeg = ExternalLibrary {
	Name = "ffmpeg",
	Depends = {
		deploy_avcodec_debug,
		deploy_avdevice_debug,
		deploy_avfilter_debug,
		deploy_avformat_debug,
		deploy_avutil_debug,
		deploy_swresample_debug,
		deploy_swscale_debug,
		deploy_avcodec_release,
		deploy_avdevice_release,
		deploy_avfilter_release,
		deploy_avformat_release,
		deploy_avutil_release,
		deploy_swresample_release,
		deploy_swscale_release,
	},
	Propagate = {
		Libs = {
			{ "avcodec.lib", "avdevice.lib", "avfilter.lib", "avformat.lib", "avutil.lib", "swresample.lib", "swscale.lib"; Config = winFilter },
			--{ "curl"; Config = linuxFilter }, -- Haven't checked linux yet
		},
	}
}

local libcurl = ExternalLibrary {
	Name = "libcurl",
	Depends = {
		deploy_libcurl_debug,
		deploy_libcurl_release,
		deploy_libssh2_debug,
		deploy_libssh2_release,
		deploy_libeay32_debug,
		deploy_libeay32_release,
		deploy_ssleay32_debug,
		deploy_ssleay32_release,
		deploy_zlib_debug,
		deploy_zlib_release,
	},
	Propagate = {
		Libs = {
			{ "libcurl_imp.lib"; Config = winFilter },
			{ "curl"; Config = linuxFilter },
		}
	}
}

local zlib = ExternalLibrary {
	Name = "zlib",
	Depends = {
		deploy_zlib_debug,
		deploy_zlib_release,
	},
	Propagate = {
		Libs = {
			{ "zlibd.lib"; Config = winDebugFilter },
			{ "zlib.lib"; Config = winReleaseFilter },
			{ "z"; Config = linuxFilter },
		}
	}
}

local lz4 = ExternalLibrary {
	Name = "lz4",
	Depends = {
		deploy_lz4,
	},
	Propagate = {
		Libs = {
			{ "lz4.lib"; Config = winFilter },
			{ "lz4"; Config = linuxFilter },
		}
	}
}

local xo = ExternalLibrary {
	Name = "xo",
	Depends = {
		deploy_xo_debug,
		deploy_xo_release,
	},
	Propagate = {
		Env = {
			LIBPATH = {
				{ "third_party/xo/t2-output/win64-msvc2015-debug-default", Config = winDebugFilter },
				{ "third_party/xo/t2-output/win64-msvc2015-release-default", Config = winReleaseFilter },
			},
		},
		Libs = { "xo.lib" },
	},
}

local tsf = StaticLibrary {
	Name = "tsf",
	Depends = { winCrt, },
	Sources = {
		"third_party/tsf/tsf.cpp",
		"third_party/tsf/tsf.h",
	},
	IdeGenerationHints = ideHintThirdParty,
}

local utfz = StaticLibrary {
	Name = "utfz",
	Depends = { winCrt, },
	Sources = {
		"third_party/utfz/utfz.cpp",
		"third_party/utfz/utfz.h",
	},
	IdeGenerationHints = ideHintThirdParty,
}

local uberlogger = Program {
	Name = "uberlogger",
	Depends = { winCrt, linuxCrt },
	Libs = {
		{ "rt"; Config = linuxFilter },
	},
	SourceDir = "third_party/uberlog",
	Sources = {
		"uberlog.cpp",
		"uberlog.h",
		"uberlogger.cpp",
		"tsf.cpp",
		"tsf.h",
	},
	IdeGenerationHints = ideHintApp,
}

local uberlog = StaticLibrary {
	Name = "uberlog",
	Depends = { winCrt, uberlogger },
	SourceDir = "third_party/uberlog",
	Sources = {
		"uberlog.cpp",
		"uberlog.h",
		--"tsf.cpp",
		--"tsf.h",
	},
	IdeGenerationHints = ideHintThirdParty,
}

local pal = SharedLibrary {
	Name = "pal",
	Depends = { winCrt, uberlog, utfz, tsf, libcurl, zlib, lz4 },
	Includes = {
		"third_party/pal",
	},
	Libs = {
		{ "Dbghelp.lib"; Config = winFilter },
		{ "uuid", "curl"; Config = linuxFilter },
	},
	PrecompiledHeader = {
		Source = "third_party/pal/pch.cpp",
		Header = "pch.h",
		Pass = "PchGen",
	},
	Sources = {
		makeGlob("third_party/pal", {}),
	},
	IdeGenerationHints = ideHintThirdParty,
}

local Annotator = Program {
	Name = "Annotator",
	Depends = {
		winCrt, xo, ffmpeg, pal, tsf,
	},
	--Env = {
	--	PROGOPTS = { "/SUBSYSTEM:CONSOLE" },
	--},
	Libs = { 
		{ "m", "stdc++"; Config = "linux-*" },
	},
	PrecompiledHeader = {
		Source = "src/pch.cpp",
		Header = "pch.h",
		Pass = "PchGen",
	},
	Includes = {
		"src", -- This is purely here for VS intellisense. With this, VS can't find pch.h from cpp files that are not in the same dir as pch.h
	},
	Sources = {
		makeGlob("src", {}),
	}
}

Default(Annotator)
