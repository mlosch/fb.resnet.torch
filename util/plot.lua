--[[
 Script to read and plot torch-optim log files.
 Usage is:
  plot.lua ylabel xlabel log1 log2 ... logN

 All graphs are plotted in the same figure.
 x and ylabel have to exist in every log file for obvious reason.
]]

require 'gnuplot'
require 'paths'

local function gsplit(str, pattern)
   local vals = {}
   for v in str:gmatch(pattern) do
      vals[#vals+1] = v
   end
   return vals
end

local function readlog(logfile, xl, yl)
   local f = io.open(logfile, 'r')
   if not f then
      print('Logfile '.. logfile ..' does not exist.')
      return nil
   end
   print('Reading logfile '.. logfile)

   local xi = 0
   local yi = 0

   local x = {}
   local y = {}

   local i = 0
   for line in f:lines() do
      local vals = gsplit(line, '%S+')
      if i == 0 then
         for k, v in ipairs(vals) do
            if v == xl then xi = k end
            if v == yl then yi = k end
         end
      else
         x[#x+1] = vals[xi]
         y[#y+1] = vals[yi]
      end
      i = i+1
   end

   return torch.Tensor(x), torch.Tensor(y)
end

------------------------------- MAIN ENTRY -------------------------------------

if #arg < 3 then
   print("Script to read and plot torch-optim log files.\
   Usage is:\
    plot.lua ylabel xlabel log1 log2 ... logN\
\
   All graphs are plotted in the same figure.\
   x and ylabel have to exist in every log file for obvious reason.")
   return 1
end

local ylabel = arg[1]
local xlabel = arg[2]

local logs = {table.unpack(arg,3)}

local plots = {}

for i=3,#arg do
   x,y = readlog(arg[i], xlabel, ylabel)
   local name = paths.basename(paths.dirname(arg[i]))
   plots[#plots+1] = {name, x, y, '-'}
end

gnuplot.plot(plots)
gnuplot.xlabel(xlabel)
gnuplot.ylabel(ylabel)
