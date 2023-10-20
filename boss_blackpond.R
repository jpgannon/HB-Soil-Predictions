#black pond should have no bedrock

# Import topo metrics
mxslope <-  raster("BlackPond/slope_rad_5m.tif")
hbtpi15 <-  raster("BlackPond/tpi15m.tif")
hbtpi200 <- raster("BlackPond/tpi200m.tif")

#Stack and rename raster grids
SPC <- stack(mxslope, hbtpi15, hbtpi200)
names(SPC) <- c('maxslope', 'tpi15', 'tpi200')

# Predict the model (this takes a long time)
map.GAM.c <- predict(SPC, modelGAMs, type = "raw", dataType = "INT1U",
                     filename = "BlackPond/BPmxslt15t200gam_new8.tif",
                     format = "GTiff", overwrite = T, progress = "text")

# Export the final predictions into a raster
writeRaster(map.GAM.c, filename = "BlackPond/bossBRprediction.tif",
            format = "GTiff", progress="text", overwrite = TRUE) 

BP <- raster("BlackPond/bossBRprediction.tif")
plot(BP)
