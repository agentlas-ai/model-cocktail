import Foundation
import Vision

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

guard CommandLine.arguments.count == 2 else {
    fail("Usage: native_ocr IMAGE")
}

let imageURL = URL(fileURLWithPath: CommandLine.arguments[1])
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

do {
    let languages = try request.supportedRecognitionLanguages()
    let preferredLanguages = ["ko-KR", "en-US"].filter { languages.contains($0) }
    if !preferredLanguages.isEmpty {
        request.recognitionLanguages = preferredLanguages
    }
    try VNImageRequestHandler(url: imageURL).perform([request])
    let lines = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
    guard !lines.isEmpty else {
        fail("Apple Vision OCR found no text")
    }
    print(lines.joined(separator: "\n"))
} catch {
    fail("Apple Vision OCR failed: \(error.localizedDescription)")
}
