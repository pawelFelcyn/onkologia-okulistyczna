export interface Detection {
    class: string;
    conf: number;
    box: [number, number, number, number];
    segments: [number, number][];
}