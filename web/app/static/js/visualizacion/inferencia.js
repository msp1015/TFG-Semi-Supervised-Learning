function gettext(text) {
    return text; // Aquí gestional la internacionalización
}

function dibujaEstadisticas(datos) {
    const matriz_confusion = datos.confusion_matrix;
    const mapaClases = JSON.parse(datos.mapa);
    const metrics = datos.average_metrics;

    // Crear la matriz de confusión
    const matrixContainer = document.getElementById('matrix-container');
    const table = document.createElement('table');

    const header = document.createElement('tr');
    const emptyHeader = document.createElement('th');
    header.appendChild(emptyHeader);

    for (let key in mapaClases) {
        const th = document.createElement('th');
        th.textContent = gettext(`Predicción ${mapaClases[key]}`);
        header.appendChild(th);
    }
    table.appendChild(header);

    Object.keys(mapaClases).forEach((key, index) => {
        const row = document.createElement('tr');

        const rowHeader = document.createElement('th');
        rowHeader.textContent = gettext(`Real ${mapaClases[key]}`);
        row.appendChild(rowHeader);

        matriz_confusion[index].forEach(value => {
            const cell = document.createElement('td');
            cell.textContent = value;
            row.appendChild(cell);
        });

        table.appendChild(row);
    });

    matrixContainer.innerHTML = ''; // Limpiar el contenedor
    matrixContainer.appendChild(table);

    // Crear las métricas
    const metricsContainer = document.getElementById('metrics-container');
    metricsContainer.innerHTML = `
        <p>${gettext('Accuracy')}: ${metrics.accuracy.toFixed(4)}</p>
        <p>${gettext('Precision')}: ${metrics.precision.toFixed(4)}</p>
        <p>${gettext('Error')}: ${metrics.error.toFixed(4)}</p>
        <p>${gettext('F1-score')}: ${metrics['f1-score'].toFixed(4)}</p>
        <p>${gettext('Recall')}: ${metrics.recall.toFixed(4)}</p>
    `;
}