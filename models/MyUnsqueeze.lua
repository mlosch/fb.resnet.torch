local MyUnsqueeze, parent = torch.class('nn.MyUnsqueeze', 'nn.Unsqueeze')

local function _assertTensor(t)
   assert(torch.isTensor(t), "This module only works on tensor")
end

function MyUnsqueeze:updateGradInput(input, gradOutput)
   _assertTensor(input)
   _assertTensor(gradOutput)
   assert(input:nElement() == gradOutput:nElement())

   self.gradInput:reshape(gradOutput, input:size())
   return self.gradInput
end
