local SpatialConvolutionT, parent = torch.class('cudnn.SpatialConvolutionT', 'cudnn.SpatialConvolution')
require 'image'

function SpatialConvolutionT:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, masterFilter)
  self.masterFilter = masterFilter
  parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  --self:reset()
end

function SpatialConvolutionT:reset(stdv)
  if self.masterFilter == nil then
    parent.reset(self, stdv)
  else
    for i=1, self.nOutputPlane do
      self.weight[i] = image.scale(self.masterFilter[i],self.kH,self.kW)
    end
  end
end
