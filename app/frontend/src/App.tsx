import { useState } from 'react';
import { Layout } from './components/Layout';
import { FileUploader } from './components/FileUploader';
import { ImageGrid } from './components/ImageGrid';
import { ActionButton } from './components/ActionButton';
import { SummaryPanel } from './components/SummaryPanel'; // Ensure this is imported
import { SegmentationViewer } from './components/SegmentationViewer';
import type { Detection } from './types/inference';

function App() {
    const [images, setImages] = useState<string[]>([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [hasResults, setHasResults] = useState(false);
    const [volume, setVolume] = useState(0);
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [detections, setDetections] = useState<Detection[]>([]);

    const handleFilesSelected = (fileList: FileList | null) => {
        if (fileList) {
            const newImages = Array.from(fileList).map((file) => URL.createObjectURL(file));
            setImages((prev) => [...prev, ...newImages]);
            setHasResults(false);
            setVolume(0);
        }
    };

    const handleMarkTumors = async () => {
        if (images.length === 0) return;
        setIsProcessing(true);
        const formData = new FormData();
        try {
            const responseWithBlob = await fetch(images[0]);
            const blob = await responseWithBlob.blob();
            formData.append('file', blob)
            const response = await fetch('http://localhost:8000/inference', {
                method: 'POST',
                body: formData,    
            });
             
            if (!response.ok) {
                throw new Error('Something went wrong during processing.');
            }

            const data = await response.json();
            console.log(data);
            setVolume(data.volume);
            setDetections(data.detections);
            setHasResults(true);
        
        } catch (error) {
            console.error("API call failed:", error);
        } finally {
            setIsProcessing(false);
        }

       

  


        // // Simulate API call
        // setTimeout(() => {
        //     setIsProcessing(false);
        //     setHasResults(true);
        //     setVolume(12.45); // 2. Set a mock volume result
        // }, 2000);
    };

    const handleRemoveImage = (indexToRemove: number) => {
        setImages(images.filter((_, index) => index !== indexToRemove));
        if (images.length <= 1) {
            setHasResults(false);
            setVolume(0);
        }
    };

    return (
        <Layout>
            {selectedImage && (
                <SegmentationViewer
                    imageUrl={selectedImage}
                    detections={detections}
                    onClose={() => setSelectedImage(null)}
                />
            )}

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-full">
                <div className="lg:col-span-8 flex flex-col gap-8">
                    <section className="space-y-4">
                        <FileUploader onFilesSelected={handleFilesSelected} />
                    </section>

                    <section className="space-y-4 min-h-[200px]">
                        {images.length > 0 && (
                            <>
                                <h3 className="text-lg font-medium text-medical-800">Scan Sequence</h3>
                                <div className="relative">
                                    <ImageGrid
                                        images={images}
                                        onRemove={handleRemoveImage}
                                        onImageClick={(src) => setSelectedImage(src)}
                                    />
                                    {hasResults && (
                                        <div className="mt-2 text-center text-sm text-accent font-medium animate-pulse">
                                            Click any image to view detailed segmentation
                                        </div>
                                    )}
                                </div>
                            </>
                        )}
                    </section>
                </div>
                <div className="lg:col-span-4 flex flex-col gap-6">
                    <div className="bg-white p-6 rounded-2xl border border-medical-200 shadow-sm sticky top-24 space-y-8">
                        <div>
                            <ActionButton
                                onClick={handleMarkTumors}
                                isLoading={isProcessing}
                                disabled={images.length === 0}
                            />
                        </div>
                        <div className="border-t border-medical-100" />
                        <div className="min-h-[300px]">
                            <SummaryPanel hasResults={hasResults} volume={volume} />
                        </div>

                    </div>
                </div>
            </div>
        </Layout>
    );
}

export default App;