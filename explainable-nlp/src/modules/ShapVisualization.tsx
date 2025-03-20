import React, { useEffect, useRef } from 'react';

interface ShapVisualizationProps {
    shapData: {
        text: string;
        values: number[];
        base_value: number;
    };
}

const ShapVisualization: React.FC<ShapVisualizationProps> = ({ shapData }) => {
    const svgRef = useRef<SVGSVGElement>(null);

    useEffect(() => {
        if (!shapData || !svgRef.current) return;

        const renderShapText = () => {
            const svgElement = svgRef.current;
            if (!svgElement) return;

            // Clear any existing content
            while (svgElement.firstChild) {
                svgElement.removeChild(svgElement.firstChild);
            }

            const { text, values } = shapData;
            const tokens = text.split(' ');

            // Calculate sizing and positioning
            const svgWidth = 800;
            const svgHeight = 150;
            const margin = { top: 40, right: 20, bottom: 40, left: 20 };

            // Set SVG dimensions
            svgElement.setAttribute('width', svgWidth.toString());
            svgElement.setAttribute('height', svgHeight.toString());
            svgElement.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);

            // Find the max absolute value for scaling
            const maxAbsValue = Math.max(...values.map(Math.abs));

            // Calculate positioning
            let currentX = margin.left;
            const baseY = svgHeight / 2;

            // Create a title
            const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            title.setAttribute('x', (svgWidth / 2).toString());
            title.setAttribute('y', '20');
            title.setAttribute('text-anchor', 'middle');
            title.setAttribute('font-weight', 'bold');
            title.textContent = 'SHAP Values Impact on Prediction';
            svgElement.appendChild(title);

            // Render each token with its SHAP value
            tokens.forEach((token, i) => {
                if (i >= values.length) return; // Safety check

                const tokenGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');

                // Calculate color intensity based on SHAP value
                const shapValue = values[i];
                const normalizedValue = shapValue / maxAbsValue; // -1 to 1

                // Determine color: red for negative, blue for positive
                const color = shapValue < 0
                    ? `rgba(255, 0, 0, ${Math.min(Math.abs(normalizedValue) * 0.8 + 0.2, 1)})`
                    : `rgba(0, 0, 255, ${Math.min(Math.abs(normalizedValue) * 0.8 + 0.2, 1)})`;

                // Create token text
                const tokenText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                tokenText.setAttribute('x', currentX.toString());
                tokenText.setAttribute('y', baseY.toString());
                tokenText.setAttribute('fill', color);
                tokenText.setAttribute('font-size', '16');
                tokenText.textContent = token;

                // Calculate token width (approximate)
                const tokenWidth = token.length * 8 + 5; // Simple approximation

                // Create SHAP value text below the token
                const valueText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                valueText.setAttribute('x', (currentX + tokenWidth / 2).toString());
                valueText.setAttribute('y', (baseY + 25).toString());
                valueText.setAttribute('text-anchor', 'middle');
                valueText.setAttribute('font-size', '12');
                valueText.textContent = shapValue.toFixed(3);

                // Add elements to group
                tokenGroup.appendChild(tokenText);
                tokenGroup.appendChild(valueText);

                // Add group to SVG
                svgElement.appendChild(tokenGroup);

                // Update x position for next token
                currentX += tokenWidth + 5;
            });

            // Add legend
            const legendY = svgHeight - 15;

            // Negative legend
            const negLegend = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            negLegend.setAttribute('x', '30');
            negLegend.setAttribute('y', legendY.toString());
            negLegend.setAttribute('fill', 'rgb(255, 0, 0)');
            negLegend.setAttribute('font-size', '12');
            negLegend.textContent = 'Negative impact';
            svgElement.appendChild(negLegend);

            // Positive legend
            const posLegend = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            posLegend.setAttribute('x', (svgWidth - 120).toString());
            posLegend.setAttribute('y', legendY.toString());
            posLegend.setAttribute('fill', 'rgb(0, 0, 255)');
            posLegend.setAttribute('font-size', '12');
            posLegend.textContent = 'Positive impact';
            svgElement.appendChild(posLegend);
        };

        renderShapText();
    }, [shapData]);

    if (!shapData) {
        return <div>No SHAP data available</div>;
    }

    return (
        <div className="shap-visualization">
            <svg ref={svgRef}></svg>
        </div>
    );
};

export default ShapVisualization;