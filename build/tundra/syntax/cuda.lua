--[[
module(..., package.seeall)

local path     = require "tundra.path"
local util     = require "tundra.util"
local boot     = require "tundra.boot"
local scanner  = require "tundra.scanner"
local depgraph = require "tundra.depgraph"

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
      Label       = 'Cuda $(@)',
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

function apply(_outer_env, options)

  _outer_env:add_setup_function(generic_asm_setup)

  _outer_env:set_many {
    ["CUDA_EXTS"] = { ".cu" },
    ["NVCCOPTS"] = "",
    ["NVCCOPTS_DEBUG"] = "",
    ["NVCCOPTS_PRODUCTION"] = "",
    ["NVCCOPTS_RELEASE"] = "",
  }
end
--]]

--[[
-- cuda.lua - Support for CUDA/nvcc

module(..., package.seeall)

local path     = require "tundra.path"

DefRule {
  Name = "CUDA",
  Command = "$(NVCC)",

  Blueprint = {
    Source = { Required = true, Type = "string" },
  },

  Setup = function (env, data)
    local src = data.Source
    local base_name = path.drop_suffix(src) 
    local objFile = "$(OBJECTDIR)$(SEP)" .. base_name .. "_cuda_" .. path.get_extension(src):sub(2) .. "$(OBJECTSUFFIX)"
    return {
      InputFiles = { src },
      OutputFiles = { objFile },
    }
  end,
}
--]]