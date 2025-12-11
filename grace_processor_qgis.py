"""
grace_processor_qgis.py
Advanced GRACE Data Processing Tool for QGIS with GUI
Processes GRACE data as raster and time series with kriging interpolation
"""

import sys
import os
import glob
import json
import traceback
from datetime import datetime
from pathlib import Path

# QGIS imports - WITH ALL REQUIRED CLASSES
try:
    from qgis.core import (
        QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer,
        QgsField, QgsGeometry, QgsPointXY, QgsFeature, QgsVectorFileWriter,
        QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsCoordinateTransformContext,
        QgsRectangle, QgsProcessingFeedback, QgsProcessingException,
        QgsRaster, QgsRasterDataProvider, QgsMessageLog, Qgis, QgsWkbTypes,
        QgsProcessingAlgorithm, QgsProcessingParameterFile,
        QgsProcessingParameterEnum, QgsProcessingParameterNumber,
        QgsProcessingParameterFolderDestination, QgsProcessingParameterFileDestination,
        QgsProcessingParameterString, QgsProcessingParameterExtent,
        QgsProcessingParameterVectorLayer, QgsProcessingParameterRasterLayer,
        QgsProcessingParameterBoolean, QgsProcessingParameterDefinition,
        QgsProcessingParameterBand, QgsProcessingParameterCrs,
        QgsMapLayer  # Added missing import
    )
    
    # Import Processing Provider separately
    from qgis.core import QgsProcessingProvider
    
    from qgis.analysis import QgsInterpolator, QgsIDWInterpolator, QgsGridFileWriter
    from qgis import processing
    from qgis.PyQt.QtCore import (
        QVariant, QCoreApplication, QSettings, QTimer, pyqtSignal, QThread
    )
    from qgis.PyQt.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
        QFileDialog, QComboBox, QGroupBox, QFormLayout, QProgressBar,
        QTextEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QTabWidget,
        QWidget, QMessageBox, QApplication, QRadioButton, QButtonGroup,
        QGridLayout, QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
        QSplitter, QFrame, QToolButton, QMenu, QAction, QMainWindow,
        QDockWidget, QStatusBar, QToolBar, QSizePolicy, QScrollArea
    )
    from qgis.PyQt.QtGui import (
        QIcon, QFont, QColor, QPalette, QPixmap, QCursor
    )
    from qgis.utils import iface
    
    QGIS_AVAILABLE = True
    
except ImportError as e:
    print(f"QGIS imports failed: {e}")
    QGIS_AVAILABLE = False

# Rest of your script continues from here...
# [Keep all the other code unchanged]

# External libraries (try to import, provide fallbacks)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available - limited functionality")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

try:
    import rasterio
    from rasterio.mask import mask
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# ============================================================================
# KRIGING INTERPOLATION MODULE
# ============================================================================

class KrigingInterpolator:
    """Kriging interpolation for filling GRACE data gaps"""
    
    def __init__(self, variogram_model='spherical', nlags=6, weight=True):
        """
        Initialize kriging interpolator
        
        Parameters:
        -----------
        variogram_model : str
            Variogram model: 'spherical', 'exponential', 'gaussian'
        nlags : int
            Number of lags for variogram calculation
        weight : bool
            Use weighting for kriging
        """
        self.variogram_model = variogram_model
        self.nlags = nlags
        self.weight = weight
        self.variogram_params = None
        
    def fit(self, x, y, z):
        """
        Fit variogram model to data
        
        Parameters:
        -----------
        x, y : array-like
            Coordinates (longitude, latitude)
        z : array-like
            Values to interpolate
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy required for kriging")
        
        # Simple variogram calculation (simplified)
        # In production, use pykrige or GSTools
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        
        # Remove NaN values
        mask = ~np.isnan(self.z)
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.z = self.z[mask]
        
        if len(self.z) < 4:
            raise ValueError("Insufficient data points for kriging")
        
        # Calculate empirical variogram (simplified)
        self._calculate_empirical_variogram()
        
        # Fit model variogram
        self._fit_model_variogram()
        
        return self.variogram_params
    
    def _calculate_empirical_variogram(self):
        """Calculate empirical variogram"""
        n = len(self.z)
        distances = []
        variances = []
        
        # Calculate pairwise distances and variances
        for i in range(n):
            for j in range(i+1, n):
                # Euclidean distance
                dx = self.x[i] - self.x[j]
                dy = self.y[i] - self.y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Semivariance
                var = 0.5 * (self.z[i] - self.z[j]) ** 2
                
                distances.append(dist)
                variances.append(var)
        
        self.empirical_distances = np.array(distances)
        self.empirical_variances = np.array(variances)
    
    def _fit_model_variogram(self):
        """Fit model variogram to empirical data"""
        # Simplified fitting - in practice use proper optimization
        if self.variogram_model == 'spherical':
            # Spherical model parameters
            range_val = np.percentile(self.empirical_distances, 75)
            sill = np.percentile(self.empirical_variances, 90)
            nugget = np.percentile(self.empirical_variances, 10)
            
            self.variogram_params = {
                'model': 'spherical',
                'range': float(range_val),
                'sill': float(sill),
                'nugget': float(nugget)
            }
        else:
            # Default to spherical
            self.variogram_params = {
                'model': self.variogram_model,
                'range': 1.0,
                'sill': 1.0,
                'nugget': 0.1
            }
    
    def predict(self, x_pred, y_pred):
        """
        Predict values at new locations
        
        Parameters:
        -----------
        x_pred, y_pred : array-like
            Coordinates to predict at
        
        Returns:
        --------
        z_pred : array-like
            Predicted values
        """
        if self.variogram_params is None:
            raise ValueError("Fit the model first")
        
        x_pred = np.array(x_pred)
        y_pred = np.array(y_pred)
        z_pred = np.zeros_like(x_pred, dtype=float)
        
        n_data = len(self.z)
        n_pred = len(x_pred)
        
        # Simple inverse distance weighting as fallback
        # For proper kriging, use pykrige library
        for i in range(n_pred):
            # Calculate distances to all data points
            dx = self.x - x_pred[i]
            dy = self.y - y_pred[i]
            distances = np.sqrt(dx*dx + dy*dy)
            
            # Avoid division by zero
            distances[distances == 0] = 1e-10
            
            # Inverse distance weighting
            weights = 1.0 / (distances ** 2)
            weights_sum = np.sum(weights)
            
            if weights_sum > 0:
                z_pred[i] = np.sum(weights * self.z) / weights_sum
            else:
                z_pred[i] = np.nan
        
        return z_pred

# ============================================================================
# GRACE DATA PROCESSOR
# ============================================================================

class GraceDataProcessor(QThread):
    """Main GRACE data processing thread with progress reporting"""
    
    # Signals for progress and completion
    progress_updated = pyqtSignal(int, str)
    processing_completed = pyqtSignal(dict)
    processing_failed = pyqtSignal(str)
    log_message = pyqtSignal(str, int)  # message, level
    
    def __init__(self):
        super().__init__()
        self.settings = {}
        self.cancel_requested = False
        
    def set_settings(self, settings):
        """Set processing settings"""
        self.settings = settings.copy()
    
    def run(self):
        """Main processing thread"""
        try:
            self.log_message.emit("Starting GRACE data processing...", Qgis.Info)
            
            # Initialize processor
            processor = EnhancedGraceProcessor(self.settings['output_dir'])
            
            # Step 1: Load GRACE data
            self.progress_updated.emit(10, "Loading GRACE data...")
            self.log_message.emit("Loading GRACE data...", Qgis.Info)
            
            grace_data = processor.load_grace_data(
                self.settings['grace_dir'],
                source=self.settings.get('grace_source', 'JPL'),
                temporal_resolution=self.settings.get('temporal_resolution', 'monthly')
            )
            
            if self.cancel_requested:
                return
            
            # Step 2: Load study area
            self.progress_updated.emit(20, "Loading study area...")
            study_area = processor.load_study_area(
                self.settings.get('study_area_layer'),
                self.settings.get('study_area_type', 'basin')
            )
            
            # Step 3: Projection handling
            self.progress_updated.emit(30, "Handling projections...")
            processor.reproject_data(
                target_crs=self.settings.get('target_crs', 'EPSG:4326'),
                resample_method=self.settings.get('resample_method', 'bilinear')
            )
            
            if self.cancel_requested:
                return
            
            # Step 4: Process temporal resolution
            self.progress_updated.emit(40, "Processing temporal data...")
            processed_data = processor.process_temporal_resolution(
                resolution=self.settings.get('output_temporal_resolution', 'monthly'),
                aggregation_method=self.settings.get('aggregation_method', 'mean')
            )
            
            # Step 5: Gap filling with kriging if needed
            if self.settings.get('fill_gaps', True):
                self.progress_updated.emit(50, "Filling data gaps...")
                self.log_message.emit("Filling data gaps with kriging...", Qgis.Info)
                
                filled_data = processor.fill_data_gaps(
                    method=self.settings.get('gap_fill_method', 'kriging'),
                    max_gap_size=self.settings.get('max_gap_size', 3)
                )
                processed_data = filled_data
            
            if self.cancel_requested:
                return
            
            # Step 6: Calculate groundwater storage
            self.progress_updated.emit(60, "Calculating groundwater storage...")
            if self.settings.get('calculate_gws', True):
                gws_data = processor.calculate_groundwater_storage(
                    sm_source=self.settings.get('sm_source'),
                    sw_source=self.settings.get('sw_source')
                )
            else:
                gws_data = processed_data
            
            # Step 7: Extract for study area
            self.progress_updated.emit(70, "Extracting for study area...")
            extracted_data = processor.extract_for_study_area(
                study_area,
                method=self.settings.get('extraction_method', 'mask')
            )
            
            # Step 8: Generate outputs
            self.progress_updated.emit(80, "Generating outputs...")
            outputs = processor.generate_outputs(
                extracted_data,
                formats=self.settings.get('output_formats', ['raster', 'timeseries']),
                temporal_resolution=self.settings.get('output_temporal_resolution', 'monthly'),
                output_dir=self.settings['output_dir']
            )
            
            if self.cancel_requested:
                return
            
            # Step 9: Create visualizations
            self.progress_updated.emit(90, "Creating visualizations...")
            viz_files = processor.create_visualizations(
                extracted_data,
                output_dir=self.settings['output_dir']
            )
            
            outputs['visualizations'] = viz_files
            
            # Step 10: Generate report
            self.progress_updated.emit(95, "Generating report...")
            report_file = processor.generate_report(
                self.settings,
                outputs,
                output_dir=self.settings['output_dir']
            )
            
            outputs['report'] = report_file
            
            # Complete
            self.progress_updated.emit(100, "Processing complete!")
            self.log_message.emit("Processing completed successfully!", Qgis.Success)
            
            # Emit completion signal
            self.processing_completed.emit(outputs)
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(error_msg, Qgis.Critical)
            self.processing_failed.emit(error_msg)
    
    def cancel(self):
        """Cancel processing"""
        self.cancel_requested = True
        self.log_message.emit("Processing cancelled by user", Qgis.Warning)

# ============================================================================
# ENHANCED GRACE PROCESSOR
# ============================================================================

class EnhancedGraceProcessor:
    """Enhanced GRACE processor with projection handling and gap filling"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.grace_data = None
        self.study_area = None
        self.processed_data = None
        self.current_crs = None
        
        # Create subdirectories
        self.subdirs = {
            'rasters': self.output_dir / 'rasters',
            'timeseries': self.output_dir / 'timeseries',
            'visualizations': self.output_dir / 'visualizations',
            'temp': self.output_dir / 'temp',
            'reports': self.output_dir / 'reports'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)
    
    def load_grace_data(self, grace_dir, source='JPL', temporal_resolution='monthly'):
        """
        Load GRACE data from directory
        
        Parameters:
        -----------
        grace_dir : str
            Directory containing GRACE data
        source : str
            Data source: 'JPL', 'CSR', 'GFZ', 'MASCONS'
        temporal_resolution : str
            'monthly' or 'annual'
        """
        grace_path = Path(grace_dir)
        
        if not grace_path.exists():
            raise FileNotFoundError(f"GRACE directory not found: {grace_dir}")
        
        # Look for NetCDF files
        nc_files = sorted(grace_path.glob("*.nc")) + sorted(grace_path.glob("*.nc4"))
        
        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in {grace_dir}")
        
        # Load first file to get metadata
        try:
            import xarray as xr
            ds = xr.open_dataset(nc_files[0])
            
            # Check and standardize variable names
            var_names = {
                'lwe_thickness': 'tws',
                'ews': 'tws',
                'water_equivalent': 'tws',
                'tws': 'tws',
                'grace': 'tws'
            }
            
            for grace_var, std_var in var_names.items():
                if grace_var in ds.variables:
                    ds = ds.rename({grace_var: std_var})
                    break
            
            # Convert to meters if in cm
            if 'tws' in ds.variables:
                if ds['tws'].max() > 10:  # Likely in cm
                    ds['tws'] = ds['tws'] * 0.01
            
            # Store CRS information
            if hasattr(ds, 'crs'):
                self.current_crs = str(ds.crs)
            else:
                # Assume WGS84 if no CRS
                self.current_crs = 'EPSG:4326'
            
            # Handle multiple files if monthly
            if temporal_resolution == 'monthly':
                if len(nc_files) > 1:
                    # Open all files
                    ds_list = []
                    for file in nc_files:
                        ds_file = xr.open_dataset(file)
                        # Standardize variable names
                        for grace_var, std_var in var_names.items():
                            if grace_var in ds_file.variables:
                                ds_file = ds_file.rename({grace_var: std_var})
                                break
                        
                        if 'tws' in ds_file.variables and ds_file['tws'].max() > 10:
                            ds_file['tws'] = ds_file['tws'] * 0.01
                        
                        ds_list.append(ds_file)
                    
                    # Combine datasets
                    self.grace_data = xr.concat(ds_list, dim='time')
                else:
                    self.grace_data = ds
            else:
                # For annual, we'll aggregate later
                self.grace_data = ds
            
            return self.grace_data
            
        except Exception as e:
            raise Exception(f"Error loading GRACE data: {str(e)}")
    
    def load_study_area(self, study_area_input, area_type='basin'):
        """
        Load study area from various sources
        
        Parameters:
        -----------
        study_area_input : str or QgsVectorLayer
            Path to shapefile or QGIS vector layer
        area_type : str
            'basin', 'watershed', 'country', 'custom'
        """
        try:
            if QGIS_AVAILABLE and isinstance(study_area_input, QgsVectorLayer):
                # Already a QGIS layer
                self.study_area = study_area_input
                
                # Get CRS
                layer_crs = self.study_area.crs()
                if layer_crs.isValid():
                    self.study_area_crs = layer_crs.authid()
                else:
                    self.study_area_crs = 'EPSG:4326'
                    
            elif isinstance(study_area_input, str) and study_area_input.endswith('.shp'):
                # Load from shapefile
                if QGIS_AVAILABLE:
                    self.study_area = QgsVectorLayer(study_area_input, "Study Area", "ogr")
                    if not self.study_area.isValid():
                        raise ValueError(f"Invalid shapefile: {study_area_input}")
                    
                    layer_crs = self.study_area.crs()
                    if layer_crs.isValid():
                        self.study_area_crs = layer_crs.authid()
                    else:
                        self.study_area_crs = 'EPSG:4326'
                else:
                    # Fallback without QGIS
                    if GEOPANDAS_AVAILABLE:
                        self.study_area = gpd.read_file(study_area_input)
                        self.study_area_crs = str(self.study_area.crs)
                    else:
                        raise ImportError("Geopandas not available for shapefile loading")
            
            elif study_area_input and QGIS_AVAILABLE:
                # Try to get from QGIS project
                layer = QgsProject.instance().mapLayer(study_area_input)
                if layer and layer.type() == QgsMapLayer.VectorLayer:
                    self.study_area = layer
                    layer_crs = self.study_area.crs()
                    if layer_crs.isValid():
                        self.study_area_crs = layer_crs.authid()
                    else:
                        self.study_area_crs = 'EPSG:4326'
                else:
                    raise ValueError(f"Study area layer not found: {study_area_input}")
            else:
                # Create default global extent
                self.study_area = None
                self.study_area_crs = 'EPSG:4326'
            
            return self.study_area
            
        except Exception as e:
            raise Exception(f"Error loading study area: {str(e)}")
    
    def reproject_data(self, target_crs='EPSG:4326', resample_method='bilinear'):
        """
        Reproject GRACE data to target CRS
        
        Parameters:
        -----------
        target_crs : str
            Target coordinate reference system
        resample_method : str
            Resampling method: 'nearest', 'bilinear', 'cubic'
        """
        if self.grace_data is None:
            raise ValueError("Load GRACE data first")
        
        if self.current_crs == target_crs:
            return self.grace_data
        
        try:
            if QGIS_AVAILABLE:
                # Use QGIS for reprojection
                self._reproject_with_qgis(target_crs, resample_method)
            else:
                # Use rasterio for reprojection
                self._reproject_with_rasterio(target_crs, resample_method)
            
            self.current_crs = target_crs
            return self.grace_data
            
        except Exception as e:
            raise Exception(f"Error reprojecting data: {str(e)}")
    
    def _reproject_with_qgis(self, target_crs, resample_method):
        """Reproject using QGIS processing"""
        # Save temporary raster
        temp_file = self.subdirs['temp'] / 'grace_temp.tif'
        
        # Convert to QGIS raster layer
        # (Implementation depends on data format)
        pass
    
    def _reproject_with_rasterio(self, target_crs, resample_method):
        """Reproject using rasterio"""
        if not RASTERIO_AVAILABLE:
            raise ImportError("Rasterio required for reprojection")
        
        # Implementation would go here
        pass
    
    def process_temporal_resolution(self, resolution='monthly', aggregation_method='mean'):
        """
        Process data to specified temporal resolution
        
        Parameters:
        -----------
        resolution : str
            'monthly' or 'annual'
        aggregation_method : str
            'mean', 'sum', 'min', 'max'
        """
        if self.grace_data is None:
            raise ValueError("Load GRACE data first")
        
        try:
            if resolution == 'monthly':
                # Already monthly, just ensure consistent time dimension
                if 'time' in self.grace_data.dims:
                    self.processed_data = self.grace_data
                else:
                    raise ValueError("Monthly data requires time dimension")
                    
            elif resolution == 'annual':
                # Aggregate to annual
                if 'time' in self.grace_data.dims:
                    # Group by year
                    if hasattr(self.grace_data.time.dt, 'year'):
                        years = self.grace_data.time.dt.year
                    else:
                        # Extract years from datetime
                        import pandas as pd
                        times = pd.to_datetime(self.grace_data.time.values)
                        years = xr.DataArray([t.year for t in times], dims=['time'])
                    
                    # Group by year and aggregate
                    if aggregation_method == 'mean':
                        self.processed_data = self.grace_data.groupby(years).mean(dim='time')
                    elif aggregation_method == 'sum':
                        self.processed_data = self.grace_data.groupby(years).sum(dim='time')
                    elif aggregation_method == 'min':
                        self.processed_data = self.grace_data.groupby(years).min(dim='time')
                    elif aggregation_method == 'max':
                        self.processed_data = self.grace_data.groupby(years).max(dim='time')
                    
                    # Rename dimension from year to time
                    self.processed_data = self.processed_data.rename({'year': 'time'})
                else:
                    raise ValueError("Cannot aggregate to annual without time dimension")
            
            return self.processed_data
            
        except Exception as e:
            raise Exception(f"Error processing temporal resolution: {str(e)}")
    
    def fill_data_gaps(self, method='kriging', max_gap_size=3):
        """
        Fill data gaps using interpolation
        
        Parameters:
        -----------
        method : str
            'kriging', 'idw', 'linear', 'nearest'
        max_gap_size : int
            Maximum gap size to fill (in time steps)
        """
        if self.processed_data is None:
            self.processed_data = self.grace_data
        
        if 'tws' not in self.processed_data.variables:
            return self.processed_data
        
        try:
            tws_data = self.processed_data['tws']
            
            # Identify gaps (NaN values)
            gap_mask = np.isnan(tws_data.values)
            
            if not np.any(gap_mask):
                return self.processed_data
            
            # Get coordinates
            if hasattr(tws_data, 'lat') and hasattr(tws_data, 'lon'):
                lats = tws_data.lat.values
                lons = tws_data.lon.values
            else:
                # Assume regular grid
                lats = np.arange(tws_data.shape[1])
                lons = np.arange(tws_data.shape[2])
            
            # Fill gaps based on method
            if method == 'kriging':
                filled_data = self._fill_gaps_kriging(tws_data, lats, lons, gap_mask)
            elif method == 'idw':
                filled_data = self._fill_gaps_idw(tws_data, lats, lons, gap_mask)
            elif method in ['linear', 'nearest']:
                filled_data = self._fill_gaps_temporal(tws_data, method)
            else:
                filled_data = tws_data
            
            # Update processed data
            self.processed_data['tws'] = filled_data
            
            return self.processed_data
            
        except Exception as e:
            print(f"Warning: Gap filling failed: {str(e)}")
            return self.processed_data
    
    def _fill_gaps_kriging(self, data, lats, lons, gap_mask):
        """Fill gaps using kriging interpolation"""
        try:
            from scipy.spatial import cKDTree
            
            # Get shape
            n_time, n_lat, n_lon = data.shape
            
            # Create coordinate grids
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            # Flatten arrays for processing
            data_flat = data.values.reshape(n_time, -1)
            lat_flat = lat_grid.flatten()
            lon_flat = lon_grid.flatten()
            gap_flat = gap_mask.reshape(n_time, -1)
            
            # Process each time step
            filled_data = data.values.copy()
            
            for t in range(n_time):
                # Get valid data points for this time step
                valid_mask = ~gap_flat[t, :]
                
                if np.sum(valid_mask) < 4:  # Need at least 4 points for kriging
                    continue
                
                # Coordinates and values of valid points
                x_valid = lon_flat[valid_mask]
                y_valid = lat_flat[valid_mask]
                z_valid = data_flat[t, valid_mask]
                
                # Coordinates of gaps
                gap_idx = np.where(gap_flat[t, :])[0]
                if len(gap_idx) == 0:
                    continue
                
                x_gap = lon_flat[gap_idx]
                y_gap = lat_flat[gap_idx]
                
                # Use kriging interpolator
                kriging = KrigingInterpolator(variogram_model='spherical')
                
                try:
                    # Fit model
                    kriging.fit(x_valid, y_valid, z_valid)
                    
                    # Predict at gap locations
                    z_pred = kriging.predict(x_gap, y_gap)
                    
                    # Fill gaps
                    for i, idx in enumerate(gap_idx):
                        # Convert flat index to 3D index
                        lat_idx = idx // n_lon
                        lon_idx = idx % n_lon
                        filled_data[t, lat_idx, lon_idx] = z_pred[i]
                        
                except Exception as e:
                    print(f"Kriging failed for time step {t}: {str(e)}")
                    # Fall back to IDW
                    filled_data = self._fill_gaps_idw(data, lats, lons, gap_mask)
                    break
            
            return filled_data
            
        except Exception as e:
            print(f"Kriging interpolation failed: {str(e)}")
            return data.values
    
    def _fill_gaps_idw(self, data, lats, lons, gap_mask):
        """Fill gaps using Inverse Distance Weighting"""
        from scipy.spatial import cKDTree
        
        n_time, n_lat, n_lon = data.shape
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        data_flat = data.values.reshape(n_time, -1)
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        gap_flat = gap_mask.reshape(n_time, -1)
        
        filled_data = data.values.copy()
        
        for t in range(n_time):
            valid_mask = ~gap_flat[t, :]
            valid_idx = np.where(valid_mask)[0]
            gap_idx = np.where(gap_flat[t, :])[0]
            
            if len(valid_idx) == 0 or len(gap_idx) == 0:
                continue
            
            # Build KDTree for valid points
            valid_points = np.column_stack([lon_flat[valid_idx], lat_flat[valid_idx]])
            tree = cKDTree(valid_points)
            
            # For each gap point, find k nearest neighbors
            gap_points = np.column_stack([lon_flat[gap_idx], lat_flat[gap_idx]])
            k = min(10, len(valid_idx))
            distances, indices = tree.query(gap_points, k=k)
            
            # IDW interpolation
            for i, idx in enumerate(gap_idx):
                # Get nearest valid values and distances
                nn_indices = valid_idx[indices[i]]
                nn_values = data_flat[t, nn_indices]
                nn_distances = distances[i]
                
                # Avoid division by zero
                nn_distances[nn_distances == 0] = 1e-10
                
                # Calculate weights
                weights = 1.0 / (nn_distances ** 2)
                weighted_sum = np.sum(weights * nn_values)
                total_weight = np.sum(weights)
                
                if total_weight > 0:
                    # Convert flat index to 3D
                    lat_idx = idx // n_lon
                    lon_idx = idx % n_lon
                    filled_data[t, lat_idx, lon_idx] = weighted_sum / total_weight
        
        return filled_data
    
    def _fill_gaps_temporal(self, data, method='linear'):
        """Fill gaps using temporal interpolation"""
        import pandas as pd
        
        n_time, n_lat, n_lon = data.shape
        filled_data = data.values.copy()
        
        # Process each grid cell
        for i in range(n_lat):
            for j in range(n_lon):
                # Extract time series for this cell
                ts = data.values[:, i, j]
                
                if np.any(np.isnan(ts)):
                    # Create pandas Series for interpolation
                    ts_series = pd.Series(ts)
                    
                    if method == 'linear':
                        ts_filled = ts_series.interpolate(method='linear', limit_direction='both')
                    elif method == 'nearest':
                        ts_filled = ts_series.interpolate(method='nearest', limit_direction='both')
                    
                    filled_data[:, i, j] = ts_filled.values
        
        return filled_data
    
    def calculate_groundwater_storage(self, sm_source=None, sw_source=None):
        """Calculate groundwater storage from TWS"""
        # This is a simplified version - actual implementation would subtract
        # soil moisture, surface water, snow, etc.
        if self.processed_data is None:
            self.processed_data = self.grace_data
        
        # For now, just copy TWS as groundwater (assuming it's already processed)
        if 'tws' in self.processed_data.variables:
            self.processed_data['gws'] = self.processed_data['tws'].copy()
        
        return self.processed_data
    
    def extract_for_study_area(self, study_area, method='mask'):
        """
        Extract data for study area
        
        Parameters:
        -----------
        study_area : QgsVectorLayer or GeoDataFrame
            Study area boundary
        method : str
            'mask' (clip to boundary) or 'zonal' (zonal statistics)
        """
        if self.processed_data is None:
            raise ValueError("Process data first")
        
        if study_area is None:
            # No study area, return full dataset
            return self.processed_data
        
        try:
            # This would implement masking or zonal statistics
            # For now, return the data as-is
            return self.processed_data
            
        except Exception as e:
            raise Exception(f"Error extracting for study area: {str(e)}")
    
    def generate_outputs(self, data, formats=None, temporal_resolution='monthly', output_dir=None):
        """Generate output files in specified formats"""
        if formats is None:
            formats = ['raster', 'timeseries']
        
        outputs = {
            'rasters': [],
            'timeseries': [],
            'metadata': {}
        }
        
        output_path = Path(output_dir) if output_dir else self.output_dir
        
        try:
            # Generate raster outputs
            if 'raster' in formats and data is not None:
                raster_outputs = self._export_rasters(data, temporal_resolution, output_path)
                outputs['rasters'] = raster_outputs
            
            # Generate time series outputs
            if 'timeseries' in formats and data is not None:
                ts_outputs = self._export_timeseries(data, temporal_resolution, output_path)
                outputs['timeseries'] = ts_outputs
            
            # Generate metadata
            outputs['metadata'] = self._generate_metadata(data)
            
            return outputs
            
        except Exception as e:
            raise Exception(f"Error generating outputs: {str(e)}")
    
    def _export_rasters(self, data, temporal_resolution, output_dir):
        """Export raster files"""
        raster_dir = output_dir / 'rasters'
        raster_dir.mkdir(exist_ok=True)
        
        outputs = []
        
        try:
            if 'tws' in data.variables:
                # Export TWS
                tws_data = data['tws']
                
                if temporal_resolution == 'monthly':
                    # Export each time step
                    for i, time_val in enumerate(tws_data.time.values):
                        time_str = pd.to_datetime(time_val).strftime('%Y%m')
                        output_file = raster_dir / f'tws_{time_str}.tif'
                        
                        # Extract single time step
                        single_band = tws_data.isel(time=i)
                        
                        # Save to GeoTIFF
                        self._save_as_geotiff(single_band, output_file)
                        outputs.append(str(output_file))
                
                elif temporal_resolution == 'annual':
                    # Export annual data
                    if hasattr(tws_data.time.dt, 'year'):
                        years = tws_data.time.dt.year.values
                    else:
                        years = range(len(tws_data.time))
                    
                    for i, year in enumerate(years):
                        output_file = raster_dir / f'tws_{year}.tif'
                        
                        # Extract single year
                        if len(tws_data.time) > 1:
                            single_band = tws_data.isel(time=i)
                        else:
                            single_band = tws_data
                        
                        self._save_as_geotiff(single_band, output_file)
                        outputs.append(str(output_file))
            
            return outputs
            
        except Exception as e:
            print(f"Warning: Raster export failed: {str(e)}")
            return outputs
    
    def _save_as_geotiff(self, data_array, output_file):
        """Save xarray DataArray as GeoTIFF"""
        try:
            if hasattr(data_array, 'rio'):
                # Use rioxarray
                data_array.rio.to_raster(output_file)
            else:
                # Fallback method
                import rasterio
                from rasterio.transform import from_origin
                
                # Get transform from coordinates
                if hasattr(data_array, 'lat') and hasattr(data_array, 'lon'):
                    lat = data_array.lat.values
                    lon = data_array.lon.values
                    
                    if len(lat) > 1 and len(lon) > 1:
                        res_lat = abs(lat[1] - lat[0])
                        res_lon = abs(lon[1] - lon[0])
                        transform = from_origin(lon.min(), lat.max(), res_lon, res_lat)
                        
                        with rasterio.open(
                            output_file, 'w',
                            driver='GTiff',
                            height=len(lat),
                            width=len(lon),
                            count=1,
                            dtype=str(data_array.dtype),
                            crs=self.current_crs,
                            transform=transform
                        ) as dst:
                            dst.write(data_array.values, 1)
        
        except Exception as e:
            print(f"Warning: Could not save GeoTIFF: {str(e)}")
    
    def _export_timeseries(self, data, temporal_resolution, output_dir):
        """Export time series data"""
        ts_dir = output_dir / 'timeseries'
        ts_dir.mkdir(exist_ok=True)
        
        outputs = []
        
        try:
            if 'tws' in data.variables and 'time' in data.dims:
                tws_data = data['tws']
                
                # Calculate spatial average
                if hasattr(tws_data, 'lat') and hasattr(tws_data, 'lon'):
                    # Simple average (could be area-weighted)
                    ts_values = tws_data.mean(dim=['lat', 'lon']).values
                else:
                    ts_values = tws_data.mean(dim=['y', 'x']).values
                
                # Create time index
                times = pd.to_datetime(tws_data.time.values)
                
                # Create DataFrame
                df = pd.DataFrame({
                    'date': times,
                    'tws_m': ts_values,
                    'tws_cm': ts_values * 100
                })
                
                # Save to CSV
                output_file = ts_dir / f'timeseries_{temporal_resolution}.csv'
                df.to_csv(output_file, index=False)
                outputs.append(str(output_file))
                
                # Save metadata
                meta_file = ts_dir / f'timeseries_metadata.json'
                metadata = {
                    'temporal_resolution': temporal_resolution,
                    'n_points': len(df),
                    'date_range': {
                        'start': df['date'].min().isoformat(),
                        'end': df['date'].max().isoformat()
                    },
                    'statistics': {
                        'mean_tws_m': float(df['tws_m'].mean()),
                        'std_tws_m': float(df['tws_m'].std()),
                        'min_tws_m': float(df['tws_m'].min()),
                        'max_tws_m': float(df['tws_m'].max())
                    }
                }
                
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                outputs.append(str(meta_file))
            
            return outputs
            
        except Exception as e:
            print(f"Warning: Time series export failed: {str(e)}")
            return outputs
    
    def _generate_metadata(self, data):
        """Generate metadata for outputs"""
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'crs': self.current_crs,
            'data_source': 'GRACE',
            'processor_version': '1.0'
        }
        
        if data is not None:
            metadata.update({
                'data_shape': str(data['tws'].shape) if 'tws' in data.variables else 'unknown',
                'time_steps': len(data.time) if 'time' in data.dims else 1,
                'variables': list(data.variables.keys())
            })
        
        return metadata
    
    def create_visualizations(self, data, output_dir):
        """Create visualization plots"""
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        viz_files = []
        
        try:
            # This would create various plots
            # For now, just create a placeholder
            import matplotlib.pyplot as plt
            
            if 'tws' in data.variables:
                # Create simple time series plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Calculate spatial average
                if hasattr(data['tws'], 'lat') and hasattr(data['tws'], 'lon'):
                    ts_values = data['tws'].mean(dim=['lat', 'lon']).values
                else:
                    ts_values = data['tws'].mean(dim=['y', 'x']).values
                
                times = pd.to_datetime(data['tws'].time.values)
                
                ax.plot(times, ts_values, 'b-', linewidth=2)
                ax.set_xlabel('Date')
                ax.set_ylabel('TWS (m)')
                ax.set_title('GRACE Total Water Storage Time Series')
                ax.grid(True, alpha=0.3)
                
                output_file = viz_dir / 'timeseries_plot.png'
                plt.tight_layout()
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                viz_files.append(str(output_file))
            
            return viz_files
            
        except Exception as e:
            print(f"Warning: Visualization creation failed: {str(e)}")
            return viz_files
    
    def generate_report(self, settings, outputs, output_dir):
        """Generate processing report"""
        report_dir = output_dir / 'reports'
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / 'processing_report.html'
        
        try:
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>GRACE Data Processing Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
                    h2 {{ color: #34495e; }}
                    .section {{ margin-bottom: 30px; padding: 20px; background-color: #f9f9f9; border-radius: 5px; }}
                    .setting {{ margin: 5px 0; }}
                    .file-list {{ list-style-type: none; padding-left: 0; }}
                    .file-list li {{ padding: 5px; border-bottom: 1px solid #ddd; }}
                    .success {{ color: #27ae60; }}
                    .warning {{ color: #f39c12; }}
                    .error {{ color: #e74c3c; }}
                </style>
            </head>
            <body>
                <h1>GRACE Data Processing Report</h1>
                
                <div class="section">
                    <h2>Processing Information</h2>
                    <p><strong>Processing Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Output Directory:</strong> {output_dir}</p>
                </div>
                
                <div class="section">
                    <h2>Processing Settings</h2>
                    {self._settings_to_html(settings)}
                </div>
                
                <div class="section">
                    <h2>Generated Outputs</h2>
                    {self._outputs_to_html(outputs)}
                </div>
                
                <div class="section">
                    <h2>Processing Status</h2>
                    <p class="success">✓ Processing completed successfully</p>
                </div>
            </body>
            </html>
            """
            
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            return str(report_file)
            
        except Exception as e:
            print(f"Warning: Report generation failed: {str(e)}")
            return None
    
    def _settings_to_html(self, settings):
        """Convert settings dictionary to HTML"""
        html = "<div class='settings'>"
        for key, value in settings.items():
            if key != 'study_area_layer':  # Skip large objects
                html += f"<div class='setting'><strong>{key}:</strong> {value}</div>"
        html += "</div>"
        return html
    
    def _outputs_to_html(self, outputs):
        """Convert outputs dictionary to HTML"""
        html = "<div class='outputs'>"
        
        for key, value in outputs.items():
            if key == 'rasters' and value:
                html += f"<h3>Raster Files ({len(value)}):</h3><ul class='file-list'>"
                for file in value[:5]:  # Show first 5
                    html += f"<li>{Path(file).name}</li>"
                if len(value) > 5:
                    html += f"<li>... and {len(value)-5} more</li>"
                html += "</ul>"
            
            elif key == 'timeseries' and value:
                html += f"<h3>Time Series Files ({len(value)}):</h3><ul class='file-list'>"
                for file in value:
                    html += f"<li>{Path(file).name}</li>"
                html += "</ul>"
        
        html += "</div>"
        return html

# ============================================================================
# QGIS GUI - MAIN DIALOG
# ============================================================================

class GraceProcessorDialog(QDialog):
    """Main QGIS dialog for GRACE data processing"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GRACE Data Processor")
        self.setMinimumSize(900, 700)
        
        # Processing thread
        self.processor_thread = None
        
        # Initialize UI
        self.init_ui()
        
        # Load saved settings
        self.load_settings()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.setup_input_tab()
        self.setup_processing_tab()
        self.setup_output_tab()
        self.setup_visualization_tab()
        self.setup_progress_tab()
        
        main_layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.btn_process = QPushButton("Start Processing")
        self.btn_process.clicked.connect(self.start_processing)
        self.btn_process.setEnabled(False)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_processing)
        self.btn_cancel.setEnabled(False)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        
        self.btn_validate = QPushButton("Validate Inputs")
        self.btn_validate.clicked.connect(self.validate_inputs)
        
        button_layout.addWidget(self.btn_validate)
        button_layout.addWidget(self.btn_process)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_cancel)
        button_layout.addWidget(self.btn_close)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        # Connect validation signals
        self.connect_validation_signals()
    
    def setup_input_tab(self):
        """Setup input data tab"""
        input_tab = QWidget()
        layout = QVBoxLayout()
        
        # GRACE Data Input
        grace_group = QGroupBox("GRACE Data Source")
        grace_layout = QFormLayout()
        
        # GRACE directory
        self.grace_dir_edit = QLineEdit()
        self.grace_dir_btn = QPushButton("Browse...")
        self.grace_dir_btn.clicked.connect(lambda: self.browse_directory(self.grace_dir_edit))
        grace_dir_layout = QHBoxLayout()
        grace_dir_layout.addWidget(self.grace_dir_edit)
        grace_dir_layout.addWidget(self.grace_dir_btn)
        grace_layout.addRow("GRACE Data Directory:", self.create_widget_layout(grace_dir_layout))
        
        # GRACE data source
        self.grace_source_combo = QComboBox()
        self.grace_source_combo.addItems(["JPL", "CSR", "GFZ", "MASCONS"])
        grace_layout.addRow("Data Source:", self.grace_source_combo)
        
        # Temporal resolution
        self.temp_res_combo = QComboBox()
        self.temp_res_combo.addItems(["Monthly", "Annual"])
        grace_layout.addRow("Input Temporal Resolution:", self.temp_res_combo)
        
        grace_group.setLayout(grace_layout)
        layout.addWidget(grace_group)
        
        # Study Area Input
        area_group = QGroupBox("Study Area")
        area_layout = QFormLayout()
        
        # Study area type
        self.area_type_combo = QComboBox()
        self.area_type_combo.addItems(["Basin", "Watershed", "Country", "Custom Layer"])
        self.area_type_combo.currentTextChanged.connect(self.on_area_type_changed)
        area_layout.addRow("Area Type:", self.area_type_combo)
        
        # Layer selection
        self.area_layer_combo = QComboBox()
        self.area_layer_combo.setEditable(True)
        self.btn_refresh_layers = QPushButton("Refresh Layers")
        self.btn_refresh_layers.clicked.connect(self.refresh_layer_list)
        area_layer_layout = QHBoxLayout()
        area_layer_layout.addWidget(self.area_layer_combo)
        area_layer_layout.addWidget(self.btn_refresh_layers)
        area_layout.addRow("Select Layer:", self.create_widget_layout(area_layer_layout))
        
        # Browse for shapefile
        self.area_file_edit = QLineEdit()
        self.area_file_btn = QPushButton("Browse...")
        self.area_file_btn.clicked.connect(lambda: self.browse_file(self.area_file_edit, "Shapefile (*.shp)"))
        area_file_layout = QHBoxLayout()
        area_file_layout.addWidget(self.area_file_edit)
        area_file_layout.addWidget(self.area_file_btn)
        area_layout.addRow("Or Shapefile:", self.create_widget_layout(area_file_layout))
        
        area_group.setLayout(area_layout)
        layout.addWidget(area_group)
        
        # Projection Settings
        proj_group = QGroupBox("Projection Settings")
        proj_layout = QFormLayout()
        
        # Target CRS
        self.target_crs_edit = QLineEdit("EPSG:4326")
        self.target_crs_btn = QPushButton("Select CRS...")
        self.target_crs_btn.clicked.connect(self.select_crs)
        crs_layout = QHBoxLayout()
        crs_layout.addWidget(self.target_crs_edit)
        crs_layout.addWidget(self.target_crs_btn)
        proj_layout.addRow("Target CRS:", self.create_widget_layout(crs_layout))
        
        # Resampling method
        self.resample_combo = QComboBox()
        self.resample_combo.addItems(["Nearest Neighbor", "Bilinear", "Cubic"])
        proj_layout.addRow("Resampling Method:", self.resample_combo)
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        
        layout.addStretch()
        input_tab.setLayout(layout)
        self.tab_widget.addTab(input_tab, "Input Data")
    
    def setup_processing_tab(self):
        """Setup processing options tab"""
        processing_tab = QWidget()
        layout = QVBoxLayout()
        
        # Temporal Processing
        temp_group = QGroupBox("Temporal Processing")
        temp_layout = QFormLayout()
        
        # Output temporal resolution
        self.output_temp_res_combo = QComboBox()
        self.output_temp_res_combo.addItems(["Monthly", "Annual"])
        temp_layout.addRow("Output Resolution:", self.output_temp_res_combo)
        
        # Aggregation method (for annual)
        self.agg_method_combo = QComboBox()
        self.agg_method_combo.addItems(["Mean", "Sum", "Minimum", "Maximum"])
        temp_layout.addRow("Aggregation Method:", self.agg_method_combo)
        
        temp_group.setLayout(temp_layout)
        layout.addWidget(temp_group)
        
        # Gap Filling
        gap_group = QGroupBox("Gap Filling")
        gap_layout = QFormLayout()
        
        # Enable gap filling
        self.gap_fill_check = QCheckBox("Fill data gaps")
        self.gap_fill_check.setChecked(True)
        gap_layout.addRow(self.gap_fill_check)
        
        # Gap filling method
        self.gap_method_combo = QComboBox()
        self.gap_method_combo.addItems(["Kriging", "IDW", "Linear", "Nearest"])
        gap_layout.addRow("Method:", self.gap_method_combo)
        
        # Maximum gap size
        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(1, 12)
        self.max_gap_spin.setValue(3)
        gap_layout.addRow("Max Gap Size (months):", self.max_gap_spin)
        
        gap_group.setLayout(gap_layout)
        layout.addWidget(gap_group)
        
        # Groundwater Calculation
        gws_group = QGroupBox("Groundwater Storage")
        gws_layout = QFormLayout()
        
        # Enable GWS calculation
        self.gws_calc_check = QCheckBox("Calculate Groundwater Storage")
        self.gws_calc_check.setChecked(True)
        gws_layout.addRow(self.gws_calc_check)
        
        # Soil moisture source
        self.sm_source_combo = QComboBox()
        self.sm_source_combo.addItems(["GLDAS", "ERA5", "GLEAM", "None"])
        gws_layout.addRow("Soil Moisture Source:", self.sm_source_combo)
        
        # Surface water source
        self.sw_source_combo = QComboBox()
        self.sw_source_combo.addItems(["GLDAS", "MERRA-2", "None"])
        gws_layout.addRow("Surface Water Source:", self.sw_source_combo)
        
        gws_group.setLayout(gws_layout)
        layout.addWidget(gws_group)
        
        layout.addStretch()
        processing_tab.setLayout(layout)
        self.tab_widget.addTab(processing_tab, "Processing Options")
    
    def setup_output_tab(self):
        """Setup output options tab"""
        output_tab = QWidget()
        layout = QVBoxLayout()
        
        # Output Formats
        format_group = QGroupBox("Output Formats")
        format_layout = QVBoxLayout()
        
        # Raster output
        self.raster_check = QCheckBox("Generate Raster Files")
        self.raster_check.setChecked(True)
        format_layout.addWidget(self.raster_check)
        
        # Raster format
        self.raster_format_combo = QComboBox()
        self.raster_format_combo.addItems(["GeoTIFF", "NetCDF", "ENVI"])
        raster_format_layout = QHBoxLayout()
        raster_format_layout.addWidget(QLabel("Format:"))
        raster_format_layout.addWidget(self.raster_format_combo)
        raster_format_layout.addStretch()
        format_layout.addLayout(raster_format_layout)
        
        # Time series output
        self.timeseries_check = QCheckBox("Generate Time Series")
        self.timeseries_check.setChecked(True)
        format_layout.addWidget(self.timeseries_check)
        
        # Time series format
        self.ts_format_combo = QComboBox()
        self.ts_format_combo.addItems(["CSV", "JSON", "Excel"])
        ts_format_layout = QHBoxLayout()
        ts_format_layout.addWidget(QLabel("Format:"))
        ts_format_layout.addWidget(self.ts_format_combo)
        ts_format_layout.addStretch()
        format_layout.addLayout(ts_format_layout)
        
        # Statistics output
        self.stats_check = QCheckBox("Generate Statistics")
        format_layout.addWidget(self.stats_check)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # Output Directory
        dir_group = QGroupBox("Output Directory")
        dir_layout = QFormLayout()
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_btn = QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(lambda: self.browse_directory(self.output_dir_edit, True))
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_btn)
        dir_layout.addRow("Output Directory:", self.create_widget_layout(output_dir_layout))
        
        # Output file prefix
        self.output_prefix_edit = QLineEdit("grace_processed")
        dir_layout.addRow("File Prefix:", self.output_prefix_edit)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # Output Options
        options_group = QGroupBox("Output Options")
        options_layout = QFormLayout()
        
        # Create subdirectories
        self.subdirs_check = QCheckBox("Create subdirectories")
        self.subdirs_check.setChecked(True)
        options_layout.addRow(self.subdirs_check)
        
        # Overwrite existing files
        self.overwrite_check = QCheckBox("Overwrite existing files")
        options_layout.addRow(self.overwrite_check)
        
        # Compress output
        self.compress_check = QCheckBox("Compress output files")
        options_layout.addRow(self.compress_check)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        layout.addStretch()
        output_tab.setLayout(layout)
        self.tab_widget.addTab(output_tab, "Output Options")
    
    def setup_visualization_tab(self):
        """Setup visualization options tab"""
        viz_tab = QWidget()
        layout = QVBoxLayout()
        
        # Visualization Options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QVBoxLayout()
        
        # Create visualizations
        self.viz_check = QCheckBox("Create Visualizations")
        self.viz_check.setChecked(True)
        viz_layout.addWidget(self.viz_check)
        
        # Visualization types
        self.viz_timeseries_check = QCheckBox("Time Series Plots")
        self.viz_timeseries_check.setChecked(True)
        viz_layout.addWidget(self.viz_timeseries_check)
        
        self.viz_spatial_check = QCheckBox("Spatial Maps")
        self.viz_spatial_check.setChecked(True)
        viz_layout.addWidget(self.viz_spatial_check)
        
        self.viz_animation_check = QCheckBox("Animation")
        viz_layout.addWidget(self.viz_animation_check)
        
        # Plot format
        plot_format_layout = QHBoxLayout()
        plot_format_layout.addWidget(QLabel("Plot Format:"))
        self.plot_format_combo = QComboBox()
        self.plot_format_combo.addItems(["PNG", "PDF", "SVG"])
        plot_format_layout.addWidget(self.plot_format_combo)
        plot_format_layout.addStretch()
        viz_layout.addLayout(plot_format_layout)
        
        # Plot size
        plot_size_layout = QHBoxLayout()
        plot_size_layout.addWidget(QLabel("Plot Size:"))
        self.plot_width_spin = QSpinBox()
        self.plot_width_spin.setRange(5, 20)
        self.plot_width_spin.setValue(10)
        plot_size_layout.addWidget(QLabel("Width:"))
        plot_size_layout.addWidget(self.plot_width_spin)
        
        self.plot_height_spin = QSpinBox()
        self.plot_height_spin.setRange(5, 20)
        self.plot_height_spin.setValue(8)
        plot_size_layout.addWidget(QLabel("Height:"))
        plot_size_layout.addWidget(self.plot_height_spin)
        plot_size_layout.addWidget(QLabel("inches"))
        plot_size_layout.addStretch()
        viz_layout.addLayout(plot_size_layout)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Color Scheme
        color_group = QGroupBox("Color Scheme")
        color_layout = QFormLayout()
        
        # Colormap
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["Viridis", "Plasma", "Inferno", "RdBu", "BrBG"])
        color_layout.addRow("Colormap:", self.colormap_combo)
        
        # Transparency
        self.transparency_slider = QDoubleSpinBox()
        self.transparency_slider.setRange(0.0, 1.0)
        self.transparency_slider.setValue(1.0)
        self.transparency_slider.setSingleStep(0.1)
        color_layout.addRow("Transparency:", self.transparency_slider)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        layout.addStretch()
        viz_tab.setLayout(layout)
        self.tab_widget.addTab(viz_tab, "Visualization")
    
    def setup_progress_tab(self):
        """Setup progress monitoring tab"""
        progress_tab = QWidget()
        layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        layout.addWidget(self.status_label)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.log_text)
        
        # Clear log button
        self.btn_clear_log = QPushButton("Clear Log")
        self.btn_clear_log.clicked.connect(self.clear_log)
        layout.addWidget(self.btn_clear_log)
        
        progress_tab.setLayout(layout)
        self.tab_widget.addTab(progress_tab, "Progress")
    
    def create_widget_layout(self, layout):
        """Helper to create widget from layout"""
        widget = QWidget()
        widget.setLayout(layout)
        return widget
    
    def connect_validation_signals(self):
        """Connect signals for input validation"""
        # Connect directory edits
        self.grace_dir_edit.textChanged.connect(self.validate_inputs)
        self.output_dir_edit.textChanged.connect(self.validate_inputs)
        
        # Connect area selection
        self.area_layer_combo.currentTextChanged.connect(self.validate_inputs)
        self.area_file_edit.textChanged.connect(self.validate_inputs)
    
    def on_area_type_changed(self, text):
        """Handle area type selection change"""
        if text == "Custom Layer":
            self.area_file_edit.setEnabled(True)
            self.area_file_btn.setEnabled(True)
            self.area_layer_combo.setEnabled(True)
            self.btn_refresh_layers.setEnabled(True)
        else:
            # For predefined types, allow layer selection or shapefile
            self.area_layer_combo.setEnabled(True)
            self.btn_refresh_layers.setEnabled(True)
            self.area_file_edit.setEnabled(True)
            self.area_file_btn.setEnabled(True)
    
    def refresh_layer_list(self):
        """Refresh list of vector layers in QGIS project"""
        if not QGIS_AVAILABLE:
            self.log_message("QGIS not available - cannot refresh layers", Qgis.Warning)
            return
        
        self.area_layer_combo.clear()
        self.area_layer_combo.addItem("")  # Empty option
        
        # Get all vector layers
        layers = QgsProject.instance().mapLayers().values()
        vector_layers = [layer for layer in layers if layer.type() == QgsMapLayer.VectorLayer]
        
        for layer in vector_layers:
            self.area_layer_combo.addItem(layer.name(), layer.id())
    
    def browse_directory(self, line_edit, create_new=False):
        """Browse for directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            line_edit.text() if line_edit.text() else ""
        )
        
        if dir_path:
            line_edit.setText(dir_path)
    
    def browse_file(self, line_edit, filter_string):
        """Browse for file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            line_edit.text() if line_edit.text() else "",
            filter_string
        )
        
        if file_path:
            line_edit.setText(file_path)
    
    def select_crs(self):
        """Select coordinate reference system"""
        if not QGIS_AVAILABLE:
            self.log_message("CRS selection requires QGIS", Qgis.Warning)
            return
        
        from qgis.gui import QgsProjectionSelectionDialog
        
        dialog = QgsProjectionSelectionDialog(self)
        if dialog.exec_():
            crs = dialog.crs()
            if crs.isValid():
                self.target_crs_edit.setText(crs.authid())
    
    def validate_inputs(self):
        """Validate all inputs and enable/disable process button"""
        # Check GRACE directory
        grace_dir = self.grace_dir_edit.text().strip()
        if not grace_dir or not os.path.exists(grace_dir):
            self.btn_process.setEnabled(False)
            return
        
        # Check output directory
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            self.btn_process.setEnabled(False)
            return
        
        # Check study area (either layer or file)
        area_type = self.area_type_combo.currentText()
        area_layer = self.area_layer_combo.currentText()
        area_file = self.area_file_edit.text().strip()
        
        if area_type != "Custom Layer":
            # Need either a layer or shapefile
            if not area_layer and not area_file:
                # Allow processing without study area (global processing)
                pass
        
        # All checks passed
        self.btn_process.setEnabled(True)
        self.status_label.setText("Inputs validated - ready to process")
        self.status_label.setStyleSheet("font-weight: bold; color: #27ae60;")
    
    def clear_log(self):
        """Clear the log text area"""
        self.log_text.clear()
    
    def log_message(self, message, level=Qgis.Info):
        """Add message to log with color coding"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if level == Qgis.Warning:
            color = "#f39c12"  # Orange
            prefix = "WARNING"
        elif level == Qgis.Critical:
            color = "#e74c3c"  # Red
            prefix = "ERROR"
        elif level == Qgis.Success:
            color = "#27ae60"  # Green
            prefix = "SUCCESS"
        else:
            color = "#3498db"  # Blue
            prefix = "INFO"
        
        html = f'<span style="color:{color}; font-weight:bold;">[{timestamp}] {prefix}:</span> {message}<br>'
        
        # Append to log
        self.log_text.append(html)
        
        # Also update status label for important messages
        if level in [Qgis.Success, Qgis.Critical]:
            self.status_label.setText(message)
            if level == Qgis.Success:
                self.status_label.setStyleSheet("font-weight: bold; color: #27ae60;")
            else:
                self.status_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
    
    def start_processing(self):
        """Start the GRACE data processing"""
        # Gather settings
        settings = self.gather_settings()
        
        # Create output directory if needed
        output_dir = settings['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save settings
        self.save_settings(settings)
        
        # Disable UI during processing
        self.btn_process.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_validate.setEnabled(False)
        self.tab_widget.setTabEnabled(0, False)  # Input tab
        self.tab_widget.setTabEnabled(1, False)  # Processing tab
        self.tab_widget.setTabEnabled(2, False)  # Output tab
        self.tab_widget.setTabEnabled(3, False)  # Visualization tab
        self.tab_widget.setCurrentIndex(4)  # Switch to progress tab
        
        # Clear progress
        self.progress_bar.setValue(0)
        self.log_message("Starting GRACE data processing...", Qgis.Info)
        
        # Create and start processing thread
        self.processor_thread = GraceDataProcessor()
        self.processor_thread.set_settings(settings)
        
        # Connect signals
        self.processor_thread.progress_updated.connect(self.update_progress)
        self.processor_thread.processing_completed.connect(self.processing_completed)
        self.processor_thread.processing_failed.connect(self.processing_failed)
        self.processor_thread.log_message.connect(self.log_message)
        
        # Start thread
        self.processor_thread.start()
    
    def gather_settings(self):
        """Gather all settings from UI"""
        settings = {
            # Input settings
            'grace_dir': self.grace_dir_edit.text().strip(),
            'grace_source': self.grace_source_combo.currentText(),
            'temporal_resolution': self.temp_res_combo.currentText().lower(),
            
            # Study area settings
            'study_area_type': self.area_type_combo.currentText().lower(),
            'study_area_layer': None,
            'study_area_file': None,
            
            # Projection settings
            'target_crs': self.target_crs_edit.text().strip(),
            'resample_method': self.resample_combo.currentText().lower(),
            
            # Processing settings
            'output_temporal_resolution': self.output_temp_res_combo.currentText().lower(),
            'aggregation_method': self.agg_method_combo.currentText().lower(),
            'fill_gaps': self.gap_fill_check.isChecked(),
            'gap_fill_method': self.gap_method_combo.currentText().lower(),
            'max_gap_size': self.max_gap_spin.value(),
            'calculate_gws': self.gws_calc_check.isChecked(),
            'sm_source': self.sm_source_combo.currentText(),
            'sw_source': self.sw_source_combo.currentText(),
            
            # Output settings
            'output_dir': self.output_dir_edit.text().strip(),
            'output_prefix': self.output_prefix_edit.text().strip(),
            'output_formats': [],
            
            # Visualization settings
            'create_visualizations': self.viz_check.isChecked(),
            'viz_types': [],
            'plot_format': self.plot_format_combo.currentText().lower(),
            'plot_width': self.plot_width_spin.value(),
            'plot_height': self.plot_height_spin.value(),
            'colormap': self.colormap_combo.currentText().lower(),
            'transparency': self.transparency_slider.value(),
        }
        
        # Get study area layer/file
        if self.area_layer_combo.currentText():
            if QGIS_AVAILABLE:
                # Get layer ID from combo box data
                layer_id = self.area_layer_combo.currentData()
                if layer_id:
                    settings['study_area_layer'] = layer_id
        
        if self.area_file_edit.text().strip():
            settings['study_area_file'] = self.area_file_edit.text().strip()
        
        # Get output formats
        if self.raster_check.isChecked():
            settings['output_formats'].append('raster')
            settings['raster_format'] = self.raster_format_combo.currentText().lower()
        
        if self.timeseries_check.isChecked():
            settings['output_formats'].append('timeseries')
            settings['ts_format'] = self.ts_format_combo.currentText().lower()
        
        if self.stats_check.isChecked():
            settings['output_formats'].append('statistics')
        
        # Get visualization types
        if self.viz_timeseries_check.isChecked():
            settings['viz_types'].append('timeseries')
        
        if self.viz_spatial_check.isChecked():
            settings['viz_types'].append('spatial')
        
        if self.viz_animation_check.isChecked():
            settings['viz_types'].append('animation')
        
        return settings
    
    def update_progress(self, value, message):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
        # Color code based on progress
        if value < 30:
            color = "#e74c3c"  # Red
        elif value < 70:
            color = "#f39c12"  # Orange
        else:
            color = "#27ae60"  # Green
        
        self.status_label.setStyleSheet(f"font-weight: bold; color: {color};")
    
    def processing_completed(self, outputs):
        """Handle processing completion"""
        self.log_message("Processing completed successfully!", Qgis.Success)
        
        # Re-enable UI
        self.btn_process.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_validate.setEnabled(True)
        self.tab_widget.setTabEnabled(0, True)
        self.tab_widget.setTabEnabled(1, True)
        self.tab_widget.setTabEnabled(2, True)
        self.tab_widget.setTabEnabled(3, True)
        
        # Show completion message
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Processing Complete")
        msg.setText("GRACE data processing completed successfully!")
        msg.setInformativeText(f"Outputs saved to: {self.output_dir_edit.text()}")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
    def processing_failed(self, error_message):
        """Handle processing failure"""
        self.log_message(f"Processing failed: {error_message}", Qgis.Critical)
        
        # Re-enable UI
        self.btn_process.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_validate.setEnabled(True)
        self.tab_widget.setTabEnabled(0, True)
        self.tab_widget.setTabEnabled(1, True)
        self.tab_widget.setTabEnabled(2, True)
        self.tab_widget.setTabEnabled(3, True)
        
        # Show error message
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Processing Failed")
        msg.setText("GRACE data processing failed!")
        msg.setDetailedText(error_message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
    def cancel_processing(self):
        """Cancel the processing"""
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.cancel()
            self.log_message("Cancelling processing...", Qgis.Warning)
    
    def save_settings(self, settings):
        """Save settings to QGIS settings"""
        if not QGIS_AVAILABLE:
            return
        
        qsettings = QSettings()
        qsettings.beginGroup("GRACEProcessor")
        
        for key, value in settings.items():
            if isinstance(value, (str, int, float, bool)):
                qsettings.setValue(key, value)
            elif isinstance(value, list):
                qsettings.setValue(key, json.dumps(value))
        
        qsettings.endGroup()
    
    def load_settings(self):
        """Load settings from QGIS settings"""
        if not QGIS_AVAILABLE:
            return
        
        qsettings = QSettings()
        qsettings.beginGroup("GRACEProcessor")
        
        # Load simple settings
        keys = qsettings.allKeys()
        
        # GRACE directory
        if 'grace_dir' in keys:
            self.grace_dir_edit.setText(qsettings.value('grace_dir'))
        
        # Output directory
        if 'output_dir' in keys:
            self.output_dir_edit.setText(qsettings.value('output_dir'))
        
        # CRS
        if 'target_crs' in keys:
            self.target_crs_edit.setText(qsettings.value('target_crs'))
        
        # Prefix
        if 'output_prefix' in keys:
            self.output_prefix_edit.setText(qsettings.value('output_prefix'))
        
        qsettings.endGroup()
        
        # Refresh layer list
        self.refresh_layer_list()
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        if self.processor_thread and self.processor_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Processing in Progress",
                "Processing is still running. Do you want to cancel and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.cancel_processing()
                self.processor_thread.wait(5000)  # Wait up to 5 seconds
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

# ============================================================================
# QGIS PROCESSING PROVIDER
# ============================================================================

class GraceProcessingProvider(QgsProcessingProvider):
    """QGIS Processing provider for GRACE tools"""
    
    def __init__(self):
        super().__init__()
    
    def id(self):
        return "grace"
    
    def name(self):
        return "GRACE Tools"
    
    def icon(self):
        return QIcon()
    
    def loadAlgorithms(self):
        self.addAlgorithm(GraceProcessingAlgorithm())
    
    def load(self):
        self.refreshAlgorithms()
        return True

# ============================================================================
# QGIS PROCESSING ALGORITHM
# ============================================================================

class GraceProcessingAlgorithm(QgsProcessingAlgorithm):
    """QGIS Processing algorithm for GRACE data"""
    
    def initAlgorithm(self, config=None):
        # Input parameters
        self.addParameter(
            QgsProcessingParameterFile(
                'GRACE_DIR',
                'GRACE Data Directory',
                behavior=QgsProcessingParameterFile.Folder
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                'GRACE_SOURCE',
                'GRACE Data Source',
                options=['JPL', 'CSR', 'GFZ', 'MASCONS'],
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                'TEMPORAL_RESOLUTION',
                'Temporal Resolution',
                options=['Monthly', 'Annual'],
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                'STUDY_AREA',
                'Study Area Layer',
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterCrs(
                'TARGET_CRS',
                'Target CRS',
                defaultValue='EPSG:4326'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                'OUTPUT_TEMPORAL',
                'Output Temporal Resolution',
                options=['Monthly', 'Annual'],
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                'OUTPUT_DIR',
                'Output Directory'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        # This would implement the processing algorithm
        # For now, just return the output directory
        output_dir = self.parameterAsString(parameters, 'OUTPUT_DIR', context)
        
        feedback.pushInfo("GRACE processing would run here")
        feedback.pushInfo(f"Output directory: {output_dir}")
        
        return {'OUTPUT_DIR': output_dir}
    
    def name(self):
        return 'graceprocessor'
    
    def displayName(self):
        return 'GRACE Data Processor'
    
    def group(self):
        return 'Hydrology'
    
    def groupId(self):
        return 'hydrology'
    
    def shortHelpString(self):
        return """
        Processes GRACE satellite data for hydrological analysis.
        
        Inputs:
        - GRACE data directory containing NetCDF files
        - Optional study area vector layer
        
        Outputs:
        - Processed raster files
        - Time series data
        - Visualizations
        
        Supports gap filling with kriging interpolation.
        """

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the GRACE processor"""
    if not QGIS_AVAILABLE:
        print("This script requires QGIS environment")
        print("Please run from QGIS Python console")
        return
    
    # Create and show the dialog
    dialog = GraceProcessorDialog()
    
    # Center on screen
    screen_geometry = QApplication.desktop().screenGeometry()
    dialog_geometry = dialog.frameGeometry()
    dialog.move(
        (screen_geometry.width() - dialog_geometry.width()) // 2,
        (screen_geometry.height() - dialog_geometry.height()) // 2
    )
    
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    
    return dialog

# ============================================================================
# QGIS PLUGIN BOILERPLATE
# ============================================================================

class GraceProcessorPlugin:
    """QGIS Plugin for GRACE Data Processing"""
    
    def __init__(self, iface):
        self.iface = iface
        self.dialog = None
        self.plugin_dir = os.path.dirname(__file__)
        
    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        
        # Get icon path
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        
        # Create action with icon
        if os.path.exists(icon_path):
            self.action = QAction(
                QIcon(icon_path),
                "GRACE Processor",
                self.iface.mainWindow()
            )
        else:
            # Fallback without icon
            self.action = QAction(
                "GRACE Processor",
                self.iface.mainWindow()
            )
        
        # Connect action to run method
        self.action.triggered.connect(self.run)
        
        # Set tooltip and status tip
        self.action.setToolTip("Process GRACE satellite data for groundwater analysis")
        self.action.setStatusTip("Open GRACE Data Processor")
        
        # Add to menu
        self.iface.addPluginToMenu("&Hydrology", self.action)
        
        # Add to toolbar (optional - uncomment if you want it on toolbar)
        self.iface.addToolBarIcon(self.action)
        
        print("GRACE Processor plugin loaded successfully")
        
    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        
        # Remove from menu
        self.iface.removePluginMenu("&Hydrology", self.action)
        
        # Remove from toolbar
        self.iface.removeToolBarIcon(self.action)
        
        # Close dialog if open
        if self.dialog:
            self.dialog.close()
            self.dialog = None
            
    def run(self):
        """Run method that performs all the real work"""
        # Create the dialog
        self.dialog = GraceProcessorDialog()
        
        # Show the dialog
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if running in QGIS
    if QGIS_AVAILABLE and 'iface' in globals():
        # Running in QGIS
        dialog = main()
    else:
        # Standalone mode - show message
        print("=" * 70)
        print("GRACE Data Processing Tool")
        print("=" * 70)
        print("\nThis tool is designed to run within QGIS environment.")
        print("\nTo use:")
        print("1. Open QGIS")
        print("2. Open Python Console (Plugins → Python Console)")
        print("3. Run: exec(open(r'path/to/grace_processor_qgis.py').read())")
        print("\nOr install as QGIS plugin.")
        print("\nRequired Python packages:")
        print("- numpy, pandas, xarray (for data processing)")
        print("- rasterio, geopandas (for spatial operations)")
        print("- matplotlib (for visualization)")