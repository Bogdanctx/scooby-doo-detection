import { Box, Button, Paper, Typography } from "@mui/material";
import { useState, type ChangeEvent } from "react";
import axios from "axios";
import type { DetectionResult } from "./interfaces";
import './UploadSection.style.css';

export default function UploadSection(
    props: {
        setDetectionResult: React.Dispatch<React.SetStateAction<DetectionResult | null>>,
    }
) {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            const file = event.target.files[0];
            setSelectedFile(file);
            props.setDetectionResult(null); // Reset results when new file is selected
            
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
            props.setDetectionResult(response.data);

            console.log('Detection result set in parent component.');
            console.log(`${JSON.stringify(response.data)}`);
            
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
        <Paper id="upload-section" elevation={3}>
            <input id="raised-button-file" accept="image/*" type="file" onChange={handleFileChange} />

            <Box id="upload-dropzone" onClick={() => document.getElementById('raised-button-file')?.click()}>
                {previewUrl ? (
                    <img src={previewUrl} id="preview-image" alt="Preview" />
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

            <Box id="upload-actions">
                <label htmlFor="raised-button-file" style={{ width: '100%' }}>
                    <Button id="select-image-button" variant="outlined" component="span" fullWidth>
                        {selectedFile ? "Change Image" : "Select Image"}
                    </Button>
                </label>
                
                {selectedFile && (
                    <Button id="detect-button" variant="contained" onClick={() => detectCharacters() } fullWidth>
                        Detect
                    </Button>
                )}
            </Box>
        </Paper>
    );
}