local ConcatCase, parent = torch.class('nn.ConcatCase', 'nn.Concat')

function ConcatCase:updateGradInput(input, gradOutput)
  if self.gradInput then
    return parent.updateGradInput(self, input, gradOutput)
  end
end

function ConcatCase:backward(input, gradOutput, scale)
  if self.gradInput then
    return parent.backward(self, input, gradOutput, scale)
  end
end
