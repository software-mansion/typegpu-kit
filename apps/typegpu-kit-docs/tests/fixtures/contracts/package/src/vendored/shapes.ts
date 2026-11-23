export function cube(size: number): { size: number } {
  return { size };
}

export function combine(...shapes: { size: number }[]): { size: number } {
  return { size: shapes.reduce((sum, shape) => sum + shape.size, 0) };
}
