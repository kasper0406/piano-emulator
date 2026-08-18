//  KeyboardView.swift — the instrument you can play with no hardware at all.
//
//  App Review 4.2.3(i) is the reason this exists and not a nicety: "your app
//  should work on its own without requiring installation of another app", and
//  container apps have been rejected under it. A piano you can only play by
//  plugging in a MIDI keyboard is a piano the reviewer cannot play.
//
//  Velocity comes from where in the key the click landed — near the pivot
//  (top) is a slow hammer, near the front (bottom) a fast one, which is what
//  the mechanism actually does. A drag across the keyboard is a glissando: the
//  gesture releases the key it leaves and strikes the one it enters.
//
//  SPDX-License-Identifier: MIT

import SwiftUI

public struct KeyboardView: View {
    /// Lowest key drawn. The compass is A0 (21) to C8 (108); four octaves fit a
    /// window, and `octaveShift` moves the window.
    public var lowestKey: Int
    public var octaves: Int
    public var onNoteOn: (UInt8, UInt8) -> Void
    public var onNoteOff: (UInt8) -> Void

    @State private var held: Int?

    public init(
        lowestKey: Int = 36, octaves: Int = 4,
        onNoteOn: @escaping (UInt8, UInt8) -> Void,
        onNoteOff: @escaping (UInt8) -> Void
    ) {
        self.lowestKey = lowestKey
        self.octaves = octaves
        self.onNoteOn = onNoteOn
        self.onNoteOff = onNoteOff
    }

    /// Semitone offsets within an octave that are black keys.
    private static let blackOffsets: Set<Int> = [1, 3, 6, 8, 10]

    private var whiteCount: Int { octaves * 7 + 1 }

    private func isBlack(_ key: Int) -> Bool {
        Self.blackOffsets.contains(((key % 12) + 12) % 12)
    }

    /// The white-key index of `key`, counting from `lowestKey`.
    private func whiteIndex(of key: Int) -> Int {
        var index = 0
        var note = lowestKey
        while note < key {
            if !isBlack(note) { index += 1 }
            note += 1
        }
        return index
    }

    private func whiteKeys() -> [Int] {
        var keys: [Int] = []
        var note = lowestKey
        while keys.count < whiteCount, note <= 108 {
            if !isBlack(note) { keys.append(note) }
            note += 1
        }
        return keys
    }

    private func blackKeys() -> [Int] {
        guard let last = whiteKeys().last else { return [] }
        return (lowestKey...last).filter(isBlack)
    }

    public var body: some View {
        GeometryReader { geometry in
            let whiteWidth = geometry.size.width / CGFloat(whiteCount)
            let blackWidth = whiteWidth * 0.62
            let blackHeight = geometry.size.height * 0.62

            ZStack(alignment: .topLeading) {
                ForEach(whiteKeys(), id: \.self) { note in
                    keyShape(
                        note, x: CGFloat(whiteIndex(of: note)) * whiteWidth, width: whiteWidth,
                        height: geometry.size.height, black: false)
                }
                ForEach(blackKeys(), id: \.self) { note in
                    keyShape(
                        note,
                        x: CGFloat(whiteIndex(of: note)) * whiteWidth - blackWidth / 2,
                        width: blackWidth, height: blackHeight, black: true)
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        let hit = keyAt(
                            point: value.location, size: geometry.size, whiteWidth: whiteWidth,
                            blackWidth: blackWidth, blackHeight: blackHeight)
                        guard let hit else { return }
                        if held != hit.key {
                            if let previous = held { onNoteOff(UInt8(previous)) }
                            held = hit.key
                            onNoteOn(UInt8(hit.key), hit.velocity)
                        }
                    }
                    .onEnded { _ in
                        if let previous = held { onNoteOff(UInt8(previous)) }
                        held = nil
                    }
            )
        }
    }

    @ViewBuilder
    private func keyShape(_ note: Int, x: CGFloat, width: CGFloat, height: CGFloat, black: Bool)
        -> some View
    {
        let down = held == note
        RoundedRectangle(cornerRadius: black ? 2 : 3)
            .fill(
                black
                    ? (down ? Color.accentColor : Color.black)
                    : (down ? Color.accentColor.opacity(0.55) : Color.white)
            )
            .overlay(
                RoundedRectangle(cornerRadius: black ? 2 : 3)
                    .stroke(Color.black.opacity(black ? 0.0 : 0.35), lineWidth: 0.5)
            )
            .frame(width: width - (black ? 0 : 1), height: height)
            .offset(x: x, y: 0)
            .zIndex(black ? 1 : 0)
            .allowsHitTesting(false)
    }

    /// Which key a point is over, and how hard it was struck. Black keys win
    /// where they overlap, which is what the eye expects.
    private func keyAt(
        point: CGPoint, size: CGSize, whiteWidth: CGFloat, blackWidth: CGFloat,
        blackHeight: CGFloat
    ) -> (key: Int, velocity: UInt8)? {
        guard point.x >= 0, point.x <= size.width, point.y >= 0, point.y <= size.height else {
            return nil
        }
        if point.y <= blackHeight {
            for note in blackKeys() {
                let x = CGFloat(whiteIndex(of: note)) * whiteWidth - blackWidth / 2
                if point.x >= x, point.x <= x + blackWidth {
                    return (note, velocity(at: point.y / blackHeight))
                }
            }
        }
        let index = Int(point.x / whiteWidth)
        let keys = whiteKeys()
        guard keys.indices.contains(index) else { return nil }
        return (keys[index], velocity(at: point.y / size.height))
    }

    /// 16 at the pivot, 127 at the front. The soft end is deliberately not 1:
    /// a mouse cannot aim, and the engine's velocity 1 is a genuine pianissimo
    /// that would sound like a mistake here.
    private func velocity(at fraction: CGFloat) -> UInt8 {
        let clamped = min(max(fraction, 0), 1)
        return UInt8(16 + clamped * 111)
    }
}
