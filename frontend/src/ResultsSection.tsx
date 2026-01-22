import type { DetectionResult } from "./interfaces";
import { Box, Typography, Paper, Grid } from "@mui/material";
import CharacterCard from "./CharacterCard";
import './ResultsSection.style.css'

export default function ResultsSection({ detectionResult }: { detectionResult: DetectionResult | null }) {    
    
    if (!detectionResult) {
        return null;
    }
    
    return (
        <Box id="detection-result">
            <Box sx={{ mb: 6 }}>
                <Typography 
                    id="detection-result-title" 
                    className="section-header" 
                    variant="h5"
                >
                    Analysis Evidence
                </Typography>
                
                <Paper id="detection-result-image" elevation={3}>
                    <img 
                        id="annotated-image" 
                        src={`data:image/png;base64,${detectionResult.detections}`} 
                        alt="Annotated Result" 
                    />
                </Paper>
            </Box>

            <Box>
                <Typography id="found-characters-title" className="section-header" variant="h5">
                    Identified Suspects ({detectionResult.patches.length})
                </Typography>
                
                <Grid container id="character-cards-grid">
                    {detectionResult.patches.length > 0 ? (
                        detectionResult.patches.map((patch, index) => (
                            <Grid key={index} display="flex">
                                <CharacterCard patch={patch} />
                            </Grid>
                        ))
                    ) : (
                        <Grid>
                            <Typography align="center" color="text.secondary" sx={{ py: 4, fontStyle: 'italic' }}>
                                Zoinks! No characters were clearly identified in this image.
                            </Typography>
                        </Grid>
                    )}
                </Grid>
            </Box>
        </Box>
    );
}