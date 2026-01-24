import { useState } from "react";
import { Layout } from "./components/Layout";
import { FileUploader } from "./components/FileUploader";
import { ImageGrid } from "./components/ImageGrid";
import { ActionButton } from "./components/ActionButton";
import { SummaryPanel } from "./components/SummaryPanel";
import { SegmentationViewer } from "./components/SegmentationViewer";

function App() {
  const [images, setImages] = useState<string[]>([]);
  const [selectedForVolume, setSelectedForVolume] = useState<Set<number>>(
    new Set(),
  );
  const [isProcessingVolume, setIsProcessingVolume] = useState(false);
  const [hasResults, setHasResults] = useState(false);
  const [volume, setVolume] = useState(0);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const handleFilesSelected = (fileList: FileList | null) => {
    if (fileList) {
      const newImages = Array.from(fileList).map((file) =>
        URL.createObjectURL(file),
      );
      const startIndex = images.length;
      setImages((prev) => [...prev, ...newImages]);
      // Auto-select newly uploaded images
      setSelectedForVolume((prev) => {
        const updated = new Set(prev);
        for (let i = 0; i < newImages.length; i++) {
          updated.add(startIndex + i);
        }
        return updated;
      });
      setHasResults(false);
      setVolume(0);
    }
  };

  const toggleImageSelection = (index: number) => {
    setSelectedForVolume((prev) => {
      const updated = new Set(prev);
      if (updated.has(index)) {
        updated.delete(index);
      } else {
        updated.add(index);
      }
      return updated;
    });
  };

  const toggleAllImages = (selectAll: boolean) => {
    if (selectAll) {
      const allIndices = new Set(images.map((_, i) => i));
      setSelectedForVolume(allIndices);
    } else {
      setSelectedForVolume(new Set());
    }
  };

  const handleMarkTumors = async () => {
    // Volume API: requires minimum 3 images selected
    if (selectedForVolume.size < 3) {
      console.warn("At least 3 images must be selected for volume calculation");
      return;
    }

    setIsProcessingVolume(true);
    const formData = new FormData();

    try {
      // Convert selected images to blobs and append to form data
      const selectedImageUrls = Array.from(selectedForVolume)
        .map((index) => images[index])
        .filter((url) => url !== undefined);

      const imageBlobs = await Promise.all(
        selectedImageUrls.map(async (imageUrl) => {
          const response = await fetch(imageUrl);
          return response.blob();
        }),
      );

      imageBlobs.forEach((blob) => {
        formData.append("files", blob);
      });

      // Call Volume API
      const response = await fetch("http://localhost:8000/volume", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Volume calculation failed");
      }

      const data = await response.json();
      console.log("Volume API Response:", data);
      setVolume(data.volume || 0);
      setHasResults(true);
    } catch (error) {
      console.error("Volume API call failed:", error);
    } finally {
      setIsProcessingVolume(false);
    }
  };

  const handleRemoveImage = (indexToRemove: number) => {
    setImages(images.filter((_, index) => index !== indexToRemove));
    setSelectedForVolume((prev) => {
      const updated = new Set(prev);
      updated.delete(indexToRemove);
      // Shift indices for remaining selections
      const shifted = new Set<number>();
      updated.forEach((idx) => {
        if (idx > indexToRemove) {
          shifted.add(idx - 1);
        } else {
          shifted.add(idx);
        }
      });
      return shifted;
    });
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
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-medium text-medical-800">
                    Scan Sequence
                  </h3>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-medical-600">
                      {selectedForVolume.size} of {images.length} selected
                    </span>
                    <button
                      onClick={() =>
                        toggleAllImages(
                          selectedForVolume.size !== images.length,
                        )
                      }
                      className="px-2 py-1 rounded bg-accent/20 hover:bg-accent/30 text-accent text-xs font-medium transition-colors"
                    >
                      {selectedForVolume.size === images.length
                        ? "Deselect All"
                        : "Select All"}
                    </button>
                    <div>
                      <h1></h1>
                    </div>
                  </div>
                </div>
                <div className="relative">
                  <ImageGrid
                    images={images}
                    selectedIndices={selectedForVolume}
                    onToggleSelect={toggleImageSelection}
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
              {selectedForVolume.size < 3 && images.length > 0 && (
                <div className="mb-4 p-4 bg-amber-50 border border-amber-200 rounded-lg">
                  <p className="text-sm text-amber-800">
                    📋 Select at least <strong>3 images</strong> for volume
                    calculation
                  </p>
                  <p className="text-xs text-amber-700 mt-1">
                    Currently selected:{" "}
                    <strong>{selectedForVolume.size}</strong> / Total:{" "}
                    <strong>{images.length}</strong>
                  </p>
                </div>
              )}
              <ActionButton
                onClick={handleMarkTumors}
                isLoading={isProcessingVolume}
                disabled={selectedForVolume.size < 3}
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
