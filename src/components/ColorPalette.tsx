import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PALETTE_COLORS } from '../types';
import './ColorPalette.css';

interface ColorPaletteProps {
    visible: boolean;
    topPx: number;
    currentColor: string;
    onSelectColor: (color: string) => void;
    onColorPickerFocus?: () => void;
    onColorPickerBlur?: () => void;
}

const ColorPalette: React.FC<ColorPaletteProps> = ({
    visible,
    topPx,
    currentColor,
    onSelectColor,
    onColorPickerFocus,
    onColorPickerBlur,
}) => {
    return (
        <AnimatePresence>
            {visible && (
                <motion.div
                    className="color-palette"
                    style={{ top: topPx + 4 }}
                    initial={{ opacity: 0, y: 8, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 4, scale: 0.97 }}
                    transition={{ type: 'spring', stiffness: 400, damping: 28 }}
                >
                    {PALETTE_COLORS.map((color) => (
                        <motion.div
                            key={color}
                            className={`color-swatch${currentColor === color ? ' active' : ''}`}
                            style={{ backgroundColor: color, color }}
                            onClick={() => onSelectColor(color)}
                            whileHover={{ scale: 1.2 }}
                            whileTap={{ scale: 0.92 }}
                        />
                    ))}
                    <div className="color-swatch-picker">
                        <input
                            type="color"
                            value={currentColor || '#5B7FBF'}
                            onChange={(e) => onSelectColor(e.target.value)}
                            onFocus={onColorPickerFocus}
                            onBlur={onColorPickerBlur}
                        />
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default React.memo(ColorPalette);
