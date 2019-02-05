imshow(im)
hold on
for nn = 1:1:size(rectGT,1)
    rectangle('position', rectGT(nn,:),'EdgeColor',rand(1,3),'LineWidth',3)
end
