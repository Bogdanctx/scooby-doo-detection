import { useState } from 'react';
import { 
    Box, 
    Typography, 
    Button, 
    Card, 
    CardMedia, 
    CardContent, 
    CardActions,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import axios from 'axios';

interface Patch {
    name: string;
    image: string; // base64 string
    detection_score: number;
    recognition_score: number;
}

const CharacterCard = ({ patch }: { patch: Patch }) => {
    const [viewState, setViewState] = useState<'idle' | 'correcting' | 'saved'>('idle');
    const [selectedLabel, setSelectedLabel] = useState<string>('');

    const sendFeedback = async (label: string) => {
        try {
            await axios.post('http://localhost:8000/api/feedback', {
                image_base64: patch.image,
                correct_label: label
            });
            setViewState('saved');
        } catch (error) {
            console.error("Error sending feedback:", error);
            alert("Failed to save feedback.");
        }
    };

    const handleSubmitCorrection = () => {
        if (selectedLabel) {
            sendFeedback(selectedLabel);
        }
    };

    const handleLabelChange = (event: SelectChangeEvent) => {
        setSelectedLabel(event.target.value as string);
    };

    return (
        <Card sx={{ width: '100%', borderRadius: '12px', boxShadow: 3 }}>
            <Box sx={{ height: 160, bgcolor: '#e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                <CardMedia
                    component="img"
                    image={`data:image/png;base64,${patch.image}`}
                    alt={patch.name}
                    sx={{ height: '100%', width: '100%', objectFit: 'cover' }}
                />
            </Box>
            <CardContent sx={{ textAlign: 'center', pb: 1 }}>
                <Typography variant="h6" component="div" sx={{ fontWeight: 'bold', color: '#1e293b' }}>
                    {patch.name}
                </Typography>
                <Typography variant="caption" display="block" sx={{ color: '#64748b' }}>
                    Detect: {(patch.detection_score * 100).toFixed(0)}% | Recog: {(patch.recognition_score * 100).toFixed(0)}%
                </Typography>
                
                {viewState === 'saved' && (
                    <Typography variant="body2" sx={{ color: 'green', fontWeight: 'bold', mt: 2 }}>
                        Thank you for your feedback!
                    </Typography>
                )}
            </CardContent>

            {viewState !== 'saved' && (
                <CardActions sx={{ flexDirection: 'column', gap: 1, padding: 2, pt: 0 }}>
                    <Box sx={{ width: '100%', borderTop: '1px solid #e2e8f0', pt: 2 }}>
                        <Typography variant="caption" sx={{ color: '#94a3b8', mb: 1, display: 'block', textAlign: 'center' }}>
                            Is this correct?
                        </Typography>
                        
                        {viewState === 'idle' ? (
                            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                                <Button 
                                    size="small" 
                                    variant="contained" 
                                    color="success" 
                                    onClick={() => sendFeedback(patch.name)}
                                    sx={{ textTransform: 'none', bgcolor: '#dcfce7', color: '#15803d', '&:hover': { bgcolor: '#bbf7d0' } }}
                                >
                                    Yes
                                </Button>
                                <Button 
                                    size="small" 
                                    variant="contained" 
                                    color="error" 
                                    onClick={() => setViewState('correcting')}
                                    sx={{ textTransform: 'none', bgcolor: '#fee2e2', color: '#b91c1c', '&:hover': { bgcolor: '#fecaca' } }}
                                >
                                    No
                                </Button>
                            </Box>
                        ) : (
                            // viewState === 'correcting'
                            <Box sx={{ width: '100%' }}>
                                <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                                    <InputLabel>Select actual name...</InputLabel>
                                    <Select
                                        value={selectedLabel}
                                        label="Select actual name..."
                                        onChange={handleLabelChange}
                                        sx={{ textAlign: 'left' }}
                                    >
                                        <MenuItem value="NOT_A_FACE" sx={{ color: 'red', fontWeight: 'bold' }}>❌ Not a face</MenuItem>
                                        <MenuItem disabled>──────────</MenuItem>
                                        <MenuItem value="Fred">Fred</MenuItem>
                                        <MenuItem value="Daphne">Daphne</MenuItem>
                                        <MenuItem value="Velma">Velma</MenuItem>
                                        <MenuItem value="Shaggy">Shaggy</MenuItem>
                                        <MenuItem value="Scooby">Scooby</MenuItem>
                                        <MenuItem value="Unknown">Unknown</MenuItem>
                                    </Select>
                                </FormControl>
                                <Box sx={{ display: 'flex', gap: 1 }}>
                                    <Button 
                                        fullWidth 
                                        variant="contained" 
                                        size="small"
                                        onClick={handleSubmitCorrection}
                                        sx={{ bgcolor: '#334155', '&:hover': { bgcolor: '#1e293b' } }}
                                    >
                                        Submit
                                    </Button>
                                    <Button 
                                        size="small"
                                        onClick={() => setViewState('idle')}
                                        sx={{ color: '#94a3b8', minWidth: 'auto' }}
                                    >
                                        Cancel
                                    </Button>
                                </Box>
                            </Box>
                        )}
                    </Box>
                </CardActions>
            )}
        </Card>
    );
};


export default CharacterCard;