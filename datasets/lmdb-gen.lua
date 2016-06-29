--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'
local lmdb = require 'lmdb'

local M = {}

local function findClasses(dir)
  local classList = {}
  local classToIdx = {}
  local keyList = {}

  local source = lmdb.env{
    Path = dir,
    Name = 'findClasses'
  }
  source:open()
  local txn = source:txn(true)
  local cursor = txn:cursor()

  local N = source:stat()['entries']
  local classidx = 1

  for i=1,N do
    local key, data = cursor:get()
    local class
    if data.Class == nil then
      -- for backwards compatability with eladhoffers lmdb generators
      class = string.split(data.Name, '_')[1]
    else
      class = data.Class
    end

    if classToIdx[class] == nil then
      table.insert(classList, class)
      classToIdx[class] = classidx
      classidx = classidx + 1
    end
    table.insert(keyList, key)

    if i < N then
      cursor:next()
    end
  end

  cursor:close()
  txn:abort()
  source:close()

   assert(#classList == 1000, 'expected 1000 ImageNet classes. Got: '.. #classList)
   return classList, classToIdx, keyList
end


function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   local trainDir = paths.concat(opt.data, 'train')
   local valDir = paths.concat(opt.data, 'val')
   assert(paths.dirp(trainDir), 'train lmdb not found: ' .. trainDir)
   assert(paths.dirp(valDir), 'val lmdb not found: ' .. valDir)

   print("=> Gathering list of images, classes and keys from trainig lmdb")
   local classList, classToIdx, trainKeyList = findClasses(trainDir)
   print(" | Gathering keys from validation lmdb")
   local _, _, valKeyList = findClasses(valDir)

   local info = {
      basedir = opt.data,
      classList = classList,
      train = {
         keys = trainKeyList,
         classToIdx = classToIdx,
      },
      val = {
         keys = valKeyList,
         classToIdx = classToIdx,
      },
   }

   print(" | saving info of databases to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
