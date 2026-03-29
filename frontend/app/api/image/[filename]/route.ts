import { NextRequest, NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import { join } from 'path';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ filename: string }> }
) {
  try {
    const { filename } = await params;

    // Sanitize hash to prevent directory traversal
    if (!filename || filename.includes('..')) {
      return NextResponse.json(
        { error: 'Invalid image filename' },
        { status: 400 }
      );
    }

    // Since this is Ore-acle Offline, we serve directly from the local data dir
    const imagePath = join(process.cwd(), '..', 'data', 'raw', 'images', filename);

    // Read the image file
    const imageBuffer = await readFile(imagePath);

    // Return the image with appropriate headers
    return new NextResponse(imageBuffer, {
      headers: {
        'Content-Type': 'image/webp',
        'Cache-Control': 'public, max-age=31536000, immutable',
      },
    });
  } catch (error) {
    console.error('Error serving image:', error);
    return NextResponse.json(
      { error: 'Image not found' },
      { status: 404 }
    );
  }
}

