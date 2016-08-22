local PrintShape, parent = torch.class('nn.PrintShape', 'nn.Module')

function PrintShape:updateOutput(input)
   local str = 'nn.PrintShape: '
   for i=1,input:dim() do
      if i > 1 then
         str = str .. ' x '
      end
      str = str .. input:size(i)
   end
   print(str)

   self.output:resizeAs(input):copy(input)
   return self.output
end
