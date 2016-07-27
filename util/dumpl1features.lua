require 'paths'
function dumpl1features(model, dir, prefix)
   prefix = prefix or ''
   --save l1 filters
   if not paths.dirp(dir) then
      paths.mkdir(dir)
   end
   if model.modules[1].modules[1].modules ~= nil then
      for i,m in ipairs(model.modules[1].modules[1].modules) do
         local file = paths.concat(dir, prefix ..'_s'.. i ..'.jpg')
         image.save(file, image.toDisplayTensor{input=m.modules[1].weight, padding=1})
      end
   else
      local file = paths.concat(dir, prefix ..'_l1.jpg')
      image.save(file, image.toDisplayTensor{input=model.modules[1].weight, padding=1})
   end
end
