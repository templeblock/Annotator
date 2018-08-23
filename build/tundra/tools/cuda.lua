module(..., package.seeall)

local path     = require "tundra.path"
local util     = require "tundra.util"
local boot     = require "tundra.boot"
local scanner  = require "tundra.scanner"
local depgraph = require "tundra.depgraph"
local platform = require "tundra.platform"
local native   = require 'tundra.native'

-- Register implicit make functions for assembly files.
-- These functions are called to transform source files in unit lists into
-- object files. This function is registered as a setup function so it will be
-- run after user modifications to the environment, but before nodes are
-- processed. This way users can override the extension lists.
local function cuda_setup(env)
  local _assemble = function(env, pass, fn)
    local object_fn = path.make_object_filename(env, fn, '$(OBJECTSUFFIX)')

    return depgraph.make_node {
      Env         = env,
      Label       = 'nvcc $(@)',
      Pass        = pass,
      Action      = "$(NVCCCOM)",
      InputFiles  = { fn },
      OutputFiles = { object_fn },
      Scanner     = scanner.make_cpp_scanner(env:get_list('CPPPATH')),
    }
  end

  for _, ext in ipairs(env:get_list("CUDA_EXTS")) do
    env:register_implicit_make_fn(ext, _assemble)
  end
end

--function apply(_outer_env, options)
--
--  _outer_env:add_setup_function(cuda_setup)
--
--  _outer_env:set_many {
--    ["CUDA_EXTS"] = { ".cu" },
--    ["NVCCOPTS"] = "",
--    ["NVCCOPTS_DEBUG"] = "",
--    ["NVCCOPTS_PRODUCTION"] = "",
--    ["NVCCOPTS_RELEASE"] = "",
--  }
--end

function apply(env, options)
  env:add_setup_function(cuda_setup)

  cuda_path = native.getenv("CUDA_PATH")

  if platform.host_platform() == "windows" then
    nvcc = '"' .. cuda_path .. '\\bin\\nvcc.exe"'
  else
    nvcc = "/usr/local/cuda/bin/nvcc"
  end

  env:set_many {
    ["CUDA_EXTS"] = { ".cu" },
    ["CUDA_PATH"] = cuda_path,
    ["NVCC"] = nvcc,
    ["NVCCOPTS"] = "",
    ["NVCCCOM"] = "$(NVCC) $(NVCCOPTS) -o $(@) -c $(<)",
  }
end

