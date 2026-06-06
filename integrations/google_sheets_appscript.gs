const CROSS_SELL_API_URL =
  PropertiesService.getScriptProperties().getProperty("CROSS_SELL_API_URL") ||
  "https://your-api-url/predict";

const SCORE_COL_HEADER = "cross_sell_score";
const PRED_COL_HEADER  = "prediction";
const HIGH_SCORE_THRESHOLD = 0.5;

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu("Cross Sell")
    .addItem("Score customers", "scoreCustomers")
    .addSeparator()
    .addItem("Clear scores", "clearScores")
    .addToUi();
}

function scoreCustomers() {
  const sheet   = SpreadsheetApp.getActiveSheet();
  const data    = sheet.getDataRange().getValues();
  const headers = data[0];
  const rows    = data.slice(1).filter(row => row.join("") !== "");

  if (rows.length === 0) {
    SpreadsheetApp.getActiveSpreadsheet().toast(
      "No data rows found.", "Cross Sell", 4
    );
    return;
  }

  SpreadsheetApp.getActiveSpreadsheet().toast(
    `Scoring ${rows.length} customers — please wait…`, "Cross Sell", 10
  );
  Logger.log(`[Cross Sell] Sending ${rows.length} records to API.`);

  const records = rows.map(row => {
    const record = {};
    headers.forEach((header, i) => { record[header] = row[i]; });
    return record;
  });

  let response;
  try {
    response = UrlFetchApp.fetch(CROSS_SELL_API_URL, {
      method: "post",
      contentType: "application/json",
      payload: JSON.stringify({ records }),
      muteHttpExceptions: true,
    });
  } catch (e) {
    SpreadsheetApp.getUi().alert("Network error: " + e.message);
    return;
  }

  if (response.getResponseCode() >= 300) {
    SpreadsheetApp.getUi().alert(
      `API error (${response.getResponseCode()}): ${response.getContentText()}`
    );
    return;
  }

  const payload = JSON.parse(response.getContentText());
  Logger.log(`[Cross Sell] Received scores for ${payload.prediction.length} records.`);

  let predCol = headers.indexOf(PRED_COL_HEADER);
  let scoreCol = headers.indexOf(SCORE_COL_HEADER);

  if (predCol === -1) {
    predCol = headers.length;
    sheet.getRange(1, predCol + 1).setValue(PRED_COL_HEADER);
  }
  if (scoreCol === -1) {
    scoreCol = headers.length + (predCol === headers.length ? 1 : 0);
    sheet.getRange(1, scoreCol + 1).setValue(SCORE_COL_HEADER);
  }

  const predValues  = [];
  const scoreValues = [];

  payload.prediction.forEach((pred, i) => {
    predValues.push([pred]);
    scoreValues.push([payload.score ? payload.score[i] : null]);
  });

  sheet.getRange(2, predCol + 1, predValues.length, 1).setValues(predValues);
  sheet.getRange(2, scoreCol + 1, scoreValues.length, 1).setValues(scoreValues);

  _highlightHighScores(sheet, rows.length, scoreCol + 1);

  if (payload.score) {
    sheet
      .getRange(2, 1, rows.length, scoreCol + 1)
      .sort({ column: scoreCol + 1, ascending: false });
  }

  SpreadsheetApp.getActiveSpreadsheet().toast(
    `Done! Scored ${rows.length} customers and sorted by propensity score.`,
    "Cross Sell",
    6
  );
  Logger.log("[Cross Sell] Scoring complete.");
}

function clearScores() {
  const sheet   = SpreadsheetApp.getActiveSheet();
  const headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];

  const predCol  = headers.indexOf(PRED_COL_HEADER);
  const scoreCol = headers.indexOf(SCORE_COL_HEADER);
  const lastRow  = sheet.getLastRow();

  if (predCol !== -1) {
    sheet.getRange(2, predCol + 1, lastRow - 1, 1).clearContent();
  }
  if (scoreCol !== -1) {
    sheet.getRange(2, scoreCol + 1, lastRow - 1, 1).clearContent();
  }

  const allRange = sheet.getRange(2, 1, lastRow - 1, sheet.getLastColumn());
  allRange.setBackground(null);

  SpreadsheetApp.getActiveSpreadsheet().toast(
    "Scores cleared and row formatting reset.", "Cross Sell", 4
  );
}

function _highlightHighScores(sheet, numRows, scoreColIndex) {
  for (let i = 0; i < numRows; i++) {
    const scoreCell = sheet.getRange(i + 2, scoreColIndex);
    const score = scoreCell.getValue();
    if (typeof score === "number" && score >= HIGH_SCORE_THRESHOLD) {
      sheet
        .getRange(i + 2, 1, 1, sheet.getLastColumn())
        .setBackground("#d9ead3");
    } else {
      sheet
        .getRange(i + 2, 1, 1, sheet.getLastColumn())
        .setBackground(null);
    }
  }
}
