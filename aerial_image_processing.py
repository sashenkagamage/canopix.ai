import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.ndimage import distance_transform_edt, generic_filter

# Load your aerial image
img = cv2.imread('plan_map_test5.png')  
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Display results
fig = plt.figure(figsize=(20, 12))

plt.subplot(3, 4, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# ==================== VEGETATION CLASSIFICATION ====================
# HSV conversion for vegetation analysis
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0 #convert to the hue image and normalize
hue = img_hsv[:, :, 0]
saturation = img_hsv[:, :, 1]
value = img_hsv[:, :, 2]

# RGB channels
R = img[:, :, 0].astype(float)
G = img[:, :, 1].astype(float)
B = img[:, :, 2].astype(float)

# NDVI-like index for vegetation health/density
ndvi = (G - R) / (G + R + 1e-6)

# Type 1: Dense tree canopy (high reflectance variation, target for canopy studies)
# HSV-based: Darker, more saturated green areas
dense_trees_hsv = ((hue >= 0.25) & (hue <= 0.45) & 
                   (saturation >= 0.10) &   # Lowered from 0.4 to catch less saturated trees
                   (value >= 0.10) & (value <= 0.85))  # Expanded range for brighter/darker trees

# RGB-based: Dense green vegetation with higher intensity
dense_trees_rgb = (G > R) & (G > B) & (G > 100) & (saturation > 0.15)

# Combine both methods
dense_trees = dense_trees_hsv | dense_trees_rgb

plt.subplot(3, 4, 2)
plt.imshow(dense_trees, cmap='Greens')
plt.title('Dense Trees (Target Zone 1)')
plt.axis('off')

# Type 2: Grassland/short vegetation (good for soil moisture, biomass)
# HSV-based detection: lighter, brighter vegetation (grass, fields, lawns)
grass_hsv = ((hue >= 0.2) & (hue <= 0.5) &  # Broader green-yellow range
             (saturation >= 0.05) &  # Very low threshold for light grass
             (value > 0.1))  # Bright areas (excludes dark tree shadows)

# RGB-based detection: G > R and G > B (catches more green vegetation)
grass_rgb = (G > R) & (G > B) & (G > 50)  # Green dominant with moderate intensity

# Combine both methods for comprehensive grass detection
grass_vegetation = grass_hsv | grass_rgb

plt.subplot(3, 4, 3)
plt.imshow(grass_vegetation, cmap='YlGn')
plt.title('Grass/Short Vegetation (Target Zone 2)')
plt.axis('off')

# Type 3: Cropland/agricultural areas (uniform texture, high biomass)
# Detect based on color uniformity
def local_std(window):
    return np.std(window)

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
texture_std = generic_filter(img_gray.astype(float), local_std, size=10)

# Low texture variation + green = likely crops/uniform vegetation
uniform_vegetation = (grass_vegetation | dense_trees) & (texture_std < 20)

plt.subplot(3, 4, 4)
plt.imshow(uniform_vegetation, cmap='summer')
plt.title('Uniform Vegetation/Crops (Target Zone 3)')
plt.axis('off')

# ==================== NDVI MAP ====================
# Vegetation health indicator (important for GNSS-R calibration)
ndvi_normalized = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min())

plt.subplot(3, 4, 5)
plt.imshow(ndvi_normalized, cmap='RdYlGn', vmin=0, vmax=1)
plt.title('NDVI-like Index (Vegetation Health)')
plt.colorbar()
plt.axis('off')

# ==================== NON-VEGETATION AREAS ====================
# Buildings/structures (these BLOCK Fresnel zones - must avoid)
gray_tolerance = 40
# Buildings: very bright, low saturation, angular features
bright_nonveg = ((np.abs(R - G) < gray_tolerance) &
                 (np.abs(G - B) < gray_tolerance) &
                 (np.abs(R - B) < gray_tolerance) &
                 ((R + G + B) > 400))  # Very bright

# Detect edges (buildings have sharp edges vs vegetation)
edges = cv2.Canny(img_gray, 50, 150)
edge_density = cv2.dilate(edges, np.ones((5,5)), iterations=1)

# Buildings = bright + many edges
buildings = bright_nonveg & (edge_density > 0)

# Clean up building mask
se = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
buildings = cv2.morphologyEx(buildings.astype(np.uint8), cv2.MORPH_CLOSE, se)

plt.subplot(3, 4, 6)
plt.imshow(buildings, cmap='Reds')
plt.title('Buildings/Structures (Signal Blockers)')
plt.axis('off')

# Roads/bare soil (not ideal targets but navigable)
roads_baresoil = bright_nonveg & ~buildings.astype(bool)

plt.subplot(3, 4, 7)
plt.imshow(roads_baresoil, cmap='gray')
plt.title('Roads/Bare Soil')
plt.axis('off')

# ==================== VEGETATION TARGET ZONES ====================
# Combine all vegetation types (these are measurement targets)
all_vegetation = dense_trees | grass_vegetation | uniform_vegetation

plt.subplot(3, 4, 8)
plt.imshow(all_vegetation, cmap='Greens')
plt.title('All Vegetation Targets')
plt.axis('off')

# ==================== FRESNEL ZONE CLEARANCE MAP ====================
# For GNSS-R, you need clear zones AROUND buildings but OVER vegetation
# Buildings must be kept away from Fresnel zones
building_exclusion = cv2.dilate(buildings, 
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

plt.subplot(3, 4, 9)
plt.imshow(building_exclusion, cmap='Reds')
plt.title('Building Exclusion Zones')
plt.axis('off')

# Valid measurement zones = vegetation areas WITHOUT building interference
valid_measurement_zones = all_vegetation & ~building_exclusion.astype(bool)

plt.subplot(3, 4, 10)
plt.imshow(valid_measurement_zones, cmap='RdYlGn')
plt.title('Valid GNSS-R Measurement Zones')
plt.axis('off')

# ==================== FLIGHT PATH ZONES ====================
# Distance transform on valid zones (find centers of large vegetation areas)
distance_from_buildings = distance_transform_edt(~building_exclusion)

# Optimal flight zones: far from buildings, over vegetation
# Weight by distance from buildings
flight_suitability = distance_from_buildings * valid_measurement_zones

plt.subplot(3, 4, 11)
plt.imshow(flight_suitability, cmap='jet')
plt.title('Flight Path Suitability\n(Distance from Obstacles)')
plt.colorbar()
plt.axis('off')

# Create clearance levels for Fresnel zone planning
# Higher clearance = better signal quality
fresnel_clearance = np.zeros_like(distance_from_buildings)
fresnel_clearance[distance_from_buildings > 10] = 1  # Minimal clearance
fresnel_clearance[distance_from_buildings > 20] = 2  # Good clearance
fresnel_clearance[distance_from_buildings > 30] = 3  # Excellent clearance

# Only show clearance over vegetation targets
fresnel_clearance_veg = fresnel_clearance * valid_measurement_zones

plt.subplot(3, 4, 12)
plt.imshow(fresnel_clearance_veg, cmap='RdYlGn', vmin=0, vmax=3)
plt.title('Fresnel Zone Clearance Quality\nOver Vegetation')
plt.colorbar(label='Clearance Level', ticks=[0,1,2,3])
plt.axis('off')

plt.tight_layout()
plt.show()

# ==================== SAVE RESULTS ====================
# Save vegetation classification maps
cv2.imwrite('dense_trees.png', dense_trees.astype(np.uint8) * 255)
cv2.imwrite('grass_vegetation.png', grass_vegetation.astype(np.uint8) * 255)
cv2.imwrite('all_vegetation_targets.png', all_vegetation.astype(np.uint8) * 255)
cv2.imwrite('buildings_obstacles.png', buildings * 255)
cv2.imwrite('building_exclusion_zones.png', building_exclusion * 255)
cv2.imwrite('valid_measurement_zones.png', valid_measurement_zones.astype(np.uint8) * 255)
cv2.imwrite('flight_suitability_map.png', 
            (flight_suitability / (flight_suitability.max() + 1e-6) * 255).astype(np.uint8))
cv2.imwrite('fresnel_clearance_levels.png', 
            (fresnel_clearance_veg / 3 * 255).astype(np.uint8))
cv2.imwrite('ndvi_map.png', (ndvi_normalized * 255).astype(np.uint8))

# Save comprehensive data for mission planning
savemat('gnss_r_mission_data.mat', {
    'dense_trees': dense_trees,
    'grass_vegetation': grass_vegetation,
    'uniform_vegetation': uniform_vegetation,
    'all_vegetation': all_vegetation,
    'ndvi_map': ndvi_normalized,
    'buildings': buildings,
    'building_exclusion_zones': building_exclusion,
    'valid_measurement_zones': valid_measurement_zones,
    'flight_suitability': flight_suitability,
    'fresnel_clearance_levels': fresnel_clearance_veg,
    'distance_from_buildings': distance_from_buildings
})

# ==================== STATISTICS ====================
total_pixels = img.shape[0] * img.shape[1]
veg_pixels = np.sum(all_vegetation)
valid_meas_pixels = np.sum(valid_measurement_zones)
building_pixels = np.sum(buildings)
tree_pixels = np.sum(dense_trees)
grass_pixels = np.sum(grass_vegetation)

print('=' * 70)
print('GNSS-R VEGETATION MEASUREMENT - IMAGE PROCESSING COMPLETE')
print('=' * 70)
print('VEGETATION TARGET AREAS:')
print(f'  Total vegetation: {veg_pixels} pixels ({100*veg_pixels/total_pixels:.1f}%)')
print(f'  - Dense trees: {tree_pixels} pixels ({100*tree_pixels/total_pixels:.1f}%)')
print(f'  - Grass/short veg: {grass_pixels} pixels ({100*grass_pixels/total_pixels:.1f}%)')
print(f'  - Uniform vegetation: {np.sum(uniform_vegetation)} pixels')
print()
print('OBSTACLES:')
print(f'  Buildings/structures: {building_pixels} pixels ({100*building_pixels/total_pixels:.1f}%)')
print(f'  Building exclusion zones: {np.sum(building_exclusion)} pixels')
print()
print('MEASUREMENT QUALITY:')
print(f'  Valid measurement zones: {valid_meas_pixels} pixels ({100*valid_meas_pixels/total_pixels:.1f}%)')
print(f'  Mean NDVI: {np.mean(ndvi_normalized[all_vegetation]):.3f}')
print(f'  Mean clearance in valid zones: {np.mean(distance_from_buildings[valid_measurement_zones]):.1f} pixels')
print()
print('FILES SAVED:')
print('  - all_vegetation_targets.png (main target map)')
print('  - valid_measurement_zones.png (cleared for GNSS-R)')
print('  - flight_suitability_map.png (optimal flight paths)')
print('  - fresnel_clearance_levels.png (signal quality zones)')
print('  - ndvi_map.png (vegetation health)')
print('  - building_exclusion_zones.png (avoid these areas)')
print('  - gnss_r_mission_data.mat (all data for QGroundControl)')
print('=' * 70)

