// Host tests for app/src/main/cpp/names.h (run with scripts/test-native.sh). Invented names, pieces and scores only.
#include "../app/src/main/cpp/names.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tw;

static int failures = 0;
#define EXPECT(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)
#define EXPECT_EQ(a, b) do { auto a_ = (a); auto b_ = (b); if (!(a_ == b_)) { \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #a); failures++; } } while (0)
static bool near(double a, double b) { return std::fabs(a - b) < 0.005; }

static Dictionary dict() {
    Dictionary d;
    // A small invented English list, padded so it counts as loaded.
    for (const char* w : {"send", "to", "pat", "tell", "well", "loop", "in", "the", "is", "this", "hi", "kashmir", "graph",
                          "india", "main", "use", "for", "neck", "can", "you", "ask", "thanks"}) d.words.insert(w);
    for (int i = 0; i < 1200; i++) d.words.insert("filler" + std::to_string(i));
    return d;
}

static void similarityMatchesTheMac() {
    // Values from the Python/Swift implementations the thresholds were tuned with.
    EXPECT(near(similarity("Patnaik", "Putnamic", true), 0.78));
    EXPECT(near(similarity("Patnaik", "Pat Nake", true), 0.93));
    EXPECT(near(similarity("Patnaik", "for neck", true), 0.39));
    EXPECT(near(similarity("Ayush", "I use", true), 0.62));
    EXPECT_EQ(sound("Podnamic", true), sound("Putnamic", true));
    EXPECT(near(ratio("abcd", "bcde"), 0.75));
}

static void terms() {
    Dictionary d = dict();
    EXPECT_EQ(nameWords("Thanks Siddhant, see Kavya's note on Grafana and the graph", d),
              (std::vector<std::string>{"Siddhant", "Kavya", "Grafana"}));
    EXPECT_EQ(termList({{"Siddhant", "AI"}, {"siddhant", "Kavya", "M4"}}), (std::vector<std::string>{"Siddhant", "Kavya"}));
    // Of many contacts, only the ones that sound like something heard.
    auto near = soundingLike({"Patnaik", "Zubin", "Mrinal", "Kavya"}, "send it to Putnamic and Ryanal", 0.6, 12);
    EXPECT_EQ(near, (std::vector<std::string>{"Patnaik", "Mrinal"}));
    EXPECT(soundingLike({"Kavya"}, "Kavya said hi", 0.6, 12).empty());   // already heard right
}

static void boosting() {
    std::vector<std::string> vocab = {kMark + "sid", "d", "hant", kMark + "sidhunt", kMark + "loop", kMark + "in"};
    auto pieces = piecesOf(vocab);
    Boost b({"Siddhant"});
    auto p1 = b.advance({}, pieces[0]);
    EXPECT(!p1.empty());
    auto p2 = b.advance(p1, pieces[1]);
    auto p3 = b.advance(p2, pieces[2]);
    EXPECT(b.complete(p3));
    EXPECT(b.advance({}, pieces[1]).empty());           // a word-internal piece can't start a name

    auto t = [](int id, int frame, float prob) { return Token{id, frame, prob}; };
    std::vector<Token> stock = {t(4, 0, 1), t(5, 30, 1), t(3, 60, 0.3f)};                 // "loop in sidhunt"
    std::vector<Token> boosted = {t(4, 0, 1), t(5, 30, 1), t(0, 60, 1), t(1, 70, 1), t(2, 80, 1)};
    int n = 0;
    auto out = acceptTerms(b, stock, boosted, pieces, &n);
    EXPECT_EQ(out.size(), size_t(5));
    EXPECT_EQ(n, 1);
    // A word the recogniser was sure of is never replaced.
    std::vector<Token> sure = {t(4, 0, 1), t(5, 30, 1), t(3, 60, 0.95f)};
    EXPECT_EQ(acceptTerms(b, sure, boosted, pieces).size(), size_t(3));
    // Half a name is not a listed term.
    EXPECT_EQ(acceptTerms(b, stock, {t(4, 0, 1), t(5, 30, 1), t(0, 60, 1), t(1, 70, 1)}, pieces).size(), size_t(3));
    // A name that began inside the word before ("Ayushpat Nayak", boosted "Patnaik" from "pat") is not swapped in,
    // or part of it would be typed twice ("Ayushpat Patnaik").
    {
        std::vector<std::string> v = {kMark + "ayushpat", kMark + "nayak", kMark + "pat", "naik"};
        auto pv = piecesOf(v);
        std::vector<Token> heard = {t(0, 0, 0.9f), t(1, 60, 0.3f)};
        EXPECT_EQ(acceptTerms(Boost({"Patnaik"}), heard, {t(2, 30, 1), t(3, 40, 1)}, pv).size(), size_t(2));
    }
    // Never into silence.
    auto silent = boosted;
    for (auto& x : silent) x.frame += 500;
    EXPECT_EQ(acceptTerms(b, stock, silent, pieces).size(), size_t(3));
}

static void correction() {
    Dictionary d = dict();
    std::vector<std::string> names = {"Ayush", "Patnaik", "Lakshmi", "TellWell"};
    auto fix = [&](std::vector<std::pair<std::string, float>> w) { return correctNames(w, names, d); };
    EXPECT_EQ(fix({{"Hi,", 0.9f}, {"this", 1}, {"is", 1}, {"Ayush", 0.9f}, {"Putnamic.", 0.3f}}), std::string("Hi, this is Ayush Patnaik."));
    EXPECT_EQ(fix({{"Ayush", 0.9f}, {"Podnamic", 0.4f}}), std::string("Ayush Patnaik"));
    EXPECT_EQ(fix({{"send", 1}, {"to", 1}, {"Pat", 0.5f}, {"Nyke", 0.3f}}), std::string("send to Patnaik"));
    EXPECT_EQ(fix({{"in", 1}, {"Kashmir.", 0.95f}}), std::string("in Kashmir."));       // heard with confidence
    EXPECT_EQ(fix({{"tell", 0.3f}, {"well", 0.3f}}), std::string("tell well"));          // English words stay
    EXPECT_EQ(correctNames({{"Putnamic", 0.2f}}, {}, d), std::string("Putnamic"));
    EXPECT_EQ(correctNames({{"Putnamic", 0.2f}}, names, Dictionary{}), std::string("Putnamic"));   // no dictionary: off
    EXPECT_EQ(spellTerms("the graph in grafana", {"Grafana"}), std::string("the graph in Grafana"));
}

int main() {
    similarityMatchesTheMac();
    terms();
    boosting();
    correction();
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("names tests passed\n");
    return 0;
}
