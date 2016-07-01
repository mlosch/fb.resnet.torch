--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  LMDB dataset loader
--

local image = require 'image'
local paths = require 'paths'
local lmdb = require 'lmdb'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local LMDBDataset = torch.class('resnet.LMDBDataset', M)

function LMDBDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.data, split)
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
   self.source = lmdb.env{
     Path = self.dir,
     Name = split,
     RDONLY = true,
     MaxReaders = opt.nThreads,
   }
  --  self.source:open()
  --  self.txn = self.source:txn(true)
   self.sourcestat = nil
   self:size()
end

function LMDBDataset:get(i)

  -- self.source:open()
  if self.txn == nil then
    self.source:open()
    self.txn = self.source:txn(true)
  end

  local data = self.txn:get(self.imageInfo.keys[i])

  -- extract image and class
  local class
  if data.Class == nil then
    -- for backwards compatibility with eladhoffers lmdb generators
    class = string.split(data.Name, '_')[1]
  else
    class = data.Class
  end
  local image = image.decompressJPG(data.Data,3,'float')

  return {
    input = image,
    target = self.imageInfo.classToIdx[class],
  }

end

function LMDBDataset:size()
  if not self.sourcestat then
    self.source:open()
    self.sourcestat = self.source:stat()
    self.source:close()
  end
   --return self.imageInfo.imageClass:size(1)
   return self.sourcestat['entries']
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function LMDBDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.RandomSizedCrop(224),
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.Lighting(0.1, pca.eigval, pca.eigvec),
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(256),
         t.ColorNormalize(meanstd),
         Crop(224),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.LMDBDataset
