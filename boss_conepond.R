# Import topo metrics
mxslope <-  raster("ConePond/slope_rad_5m.tif")
hbtpi15 <-  raster("ConePond/tpi15m.tif")
hbtpi200 <- raster("ConePond/tpi200m.tif")

#Stack and rename raster grids
SPC <- stack(mxslope, hbtpi15, hbtpi200)
names(SPC) <- c('maxslope', 'tpi15', 'tpi200')

# Predict the model (this takes a long time)
map.GAM.c <- predict(SPC, modelGAMs, type = "raw", dataType = "INT1U",
                     filename = "ConePond/CPmxslt15t200gam_new8.tif",
                     format = "GTiff", overwrite = T, progress = "text")

# Export the final predictions into a raster
writeRaster(map.GAM.c, filename = "ConePond/bossBRprediction.tif",
            format = "GTiff", progress="text", overwrite = TRUE) 

CP <- raster("ConePond/bossBRprediction.tif")
plot(CP)
