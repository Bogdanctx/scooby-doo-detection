import { useState } from 'react'
import { 
    Box, 
    Typography, 
    Button, 
    Paper, 
    Grid
} from '@mui/material'
import type { ChangeEvent } from 'react'
import axios from 'axios'
import './App.css'
import type { DetectionResult } from './interfaces'
import CharacterCard from './CharacterCard'

function App() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);

    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            const file = event.target.files[0];
            setSelectedFile(file);
            setDetectionResult(null); // Reset results when new file is selected
            
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreviewUrl(reader.result as string);
            };
            reader.readAsDataURL(file);
        }
    };

    const detectCharacters = async () => {
        if (!selectedFile) {
            return;
        }

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await axios.post('http://localhost:8000/api/detect', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            console.log('Detection result:', response.data);
            setDetectionResult(response.data);
            
            // Auto scroll to results
            setTimeout(() => {
                document.getElementById('detection-result')?.scrollIntoView({ behavior: 'smooth' });
            }, 100);

        } catch (error) {
            console.error('Error during detection:', error);
            alert("Error during detection. Check backend console.");
        }
    }

	return (
		<>
			{/* Header */}
			<Box id="header" sx={{ width: '100%', textAlign: 'center', padding: '1rem' }}>
				<Typography variant="h2" align="center" sx={{ fontSize: '2rem', color: 'rgb(15, 118, 110)' }}>
					Scooby-Doo Detection
				</Typography>
				<Typography variant="h5" sx={{ fontSize: '0.8rem', color: 'rgb(71, 85, 105)' }}>
					Upload an image to identify the Mystery Inc. gang!
				</Typography>
			</Box>

			{/* Main Content */}
			<Box id="main-content" sx={{ width: '100%', pb: 8 }}>
                {/* Upload Section */}
                <Paper
                    elevation={3}
                    sx={{
                        padding: '2rem',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: '1.5rem',
                        maxWidth: '600px',
                        width: '100%',
                        borderRadius: '12px',
                        margin: '0 auto',
                        mb: 4
                    }}
                >
                    <input
                        accept="image/*"
                        style={{ display: 'none' }}
                        id="raised-button-file"
                        type="file"
                        onChange={handleFileChange}
                    />

                    <Box
                        sx={{
                            width: '100%',
                            height: '300px',
                            border: '2px dashed rgb(203, 213, 225)',
                            borderRadius: '8px',
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            overflow: 'hidden',
                            backgroundColor: 'rgb(248, 250, 252)',
                            cursor: 'pointer'
                        }}
                        onClick={() => document.getElementById('raised-button-file')?.click()}
                    >
                        {previewUrl ? (
                            <img src={previewUrl} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                        ) : (
                            <Box sx={{ textAlign: 'center', p: 2 }}>
                                <Typography variant="body1" sx={{ color: 'rgb(100, 116, 139)' }}>
                                    Click to select an image
                                </Typography>
                                <Typography variant="caption" sx={{ color: 'rgb(148, 163, 184)' }}>
                                    JPG, PNG supported
                                </Typography>
                            </Box>
                        )}
                    </Box>

                    <Box sx={{ display: 'flex', gap: 2, width: '100%' }}>
                        <label htmlFor="raised-button-file" style={{ width: '100%' }}>
                            <Button 
                                variant="outlined" 
                                component="span"
                                fullWidth
                                sx={{ 
                                    color: 'rgb(15, 118, 110)', 
                                    borderColor: 'rgb(15, 118, 110)',
                                    '&:hover': { borderColor: 'rgb(13, 100, 90)', backgroundColor: 'rgba(15, 118, 110, 0.04)' }
                                }}
                            >
                                {selectedFile ? "Change Image" : "Select Image"}
                            </Button>
                        </label>
                        
                        {selectedFile && (
                            <Button 
                                variant="contained" 
                                fullWidth
                                sx={{ 
                                    backgroundColor: 'rgb(15, 118, 110)',
                                    '&:hover': { backgroundColor: 'rgb(13, 100, 90)' }
                                }}
                                onClick={detectCharacters}
                            >
                                Detect
                            </Button>
                        )}
                    </Box>
                </Paper>

                {/* --- RESULTS SECTION --- */}
                {detectionResult && detectionResult.patches.length > 0 && (
                    <Box id="detection-result" sx={{ maxWidth: '1000px', margin: '0 auto', px: 2 }}>
                        
                        {/* Annotated Image */}
                        <Box sx={{ mb: 6 }}>
                            <Typography variant="h4" sx={{ mb: 2, color: '#0f766e', fontWeight: 'bold', borderLeft: '4px solid #0f766e', pl: 2 }}>
                                Detection Result
                            </Typography>
                            <Paper elevation={2} sx={{ p: 2, bgcolor: '#f8fafc', display: 'flex', justifyContent: 'center' }}>
                                <img 
                                    src={`data:image/png;base64,${detectionResult.detections}`} 
                                    alt="Annotated Result" 
                                    style={{ maxWidth: '100%', maxHeight: '600px', borderRadius: '4px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} 
                                />
                            </Paper>
                        </Box>

                        {/* Found Characters Grid */}
                        <Box>
                            <Typography variant="h4" sx={{ mb: 2, color: '#ea580c', fontWeight: 'bold', borderLeft: '4px solid #ea580c', pl: 2 }}>
                                Found Characters
                            </Typography>
                            
                            <Grid container spacing={2}>
                                {detectionResult.patches.map((patch, index) => (
                                    <Grid key={index} display={"flex"} justifyContent="center">
                                        <CharacterCard patch={patch} />
                                    </Grid>
                                ))}
                            </Grid>
                        </Box>
                    </Box>
                )} : { detectionResult?.patches.length === 0 && (
                    <Typography variant="body1" align="center" sx={{ color: '#64748b', fontStyle: 'italic', mt: 4 }}>
                        No characters were detected in this image.
                    </Typography>
                )}
			</Box>

			{/* Footer */}
			<Box 
				id="footer"
				sx={{
					width: '100%',
					textAlign: 'center',
					padding: '1rem',
					backgroundColor: 'rgb(15, 118, 110)',
					color: 'white',
					position: 'fixed',
					bottom: 0,
                    zIndex: 10
				}}
			>
				<Typography variant="body1" align="center" sx={{ fontSize: '0.9rem' }}>
					&copy; 2024 Scooby-Doo Detection. All rights reserved.
				</Typography>
			</Box>
		</>
	)
}

export default App