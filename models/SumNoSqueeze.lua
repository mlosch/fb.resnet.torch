local SumNoSqueeze, parent = torch.class('nn.SumNoSqueeze', 'nn.Sum')

function SumNoSqueeze:__init(dimension, nInputDims, sizeAverage)
   parent.__init(self, dimension, nInputDims, sizeAverage)
end

function SumNoSqueeze:updateOutput(input)
    local dimension = self:_getPositiveDimension(input)
    if type(self.output) == 'number' then
        self.output = input.new()
    end
    self.output:sum(input, dimension)
    if self.sizeAverage then
        self.output:div(input:size(dimension))
    end
    return self.output
end

-- function SumNoSqueeze:updateGradInput(input, gradOutput)
--     local dimension = self:_getPositiveDimension(input)
--     -- zero-strides dont work with MKL/BLAS, so
--     -- dont set self.gradInput to zero-stride tensor.
--     -- Instead, do a deepcopy
--     local size      = input:size()
--     size[dimension] = 1
--     if not gradOutput:isContiguous() then
--         self._gradOutput = self._gradOutput or gradOutput.new()
--         self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
--         gradOutput = self._gradOutput
--     end
--     gradOutput      = gradOutput:view(size)
--     self.gradInput:resizeAs(input)
--     self.gradInput:copy(gradOutput:expandAs(input))
--     if self.sizeAverage then
--         self.gradInput:div(input:size(dimension))
--     end
--     return self.gradInput
-- end
