export function drawBoxes(predictions, canvas, clear = true) {
    const ctx = canvas.getContext("2d");
    if (clear) ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const font = "15px ariel";
    ctx.font = font;
    ctx.textBaseline = "top";

    const boxColor = "#22cc22";

    predictions.forEach(prediction => {
        let [x, y, width, height] = prediction.bbox

        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        var label = (prediction.score * 100).toFixed(1) + "%"
        ctx.fillStyle = boxColor;
        const textWidth = ctx.measureText(label).width + 2;
        const textHeight = parseInt(font, 10) + 4

        ctx.fillRect(x, y - 2, textWidth, textHeight);
        ctx.fillRect(x, y + height - textHeight, textWidth, textHeight);

        ctx.fillStyle = "#ffff";
        ctx.fillText(prediction.class, x, y);
        ctx.fillText(label, x, y + height - textHeight);
    });
}
