/* 
* *
* * Filename: BCJRDecoder.hpp
* *
* * Description:
* *   BCJR (Bahl-Cocke-Jelinek-Raviv) MAP decoder for the CCSDS-standard
* *   K=7, Rate 1/2 inner convolutional code (same trellis as the encoder).
* *  - Max-Log-MAP forward/backward recursions over the 64-state trellis.
* *  - Coded LLR convention L=log P(bit=0)/P(bit=1) (L>0 favors bit 0).
* *  - Free non-terminated endpoint: start state known (0), end state free.
* *  - C2 carries the CCSDS 131.0-B-5 G2 output inversion to mirror the encoder.
* *  - Emits hard decisions plus a-posteriori (APP) LLRs per info bit.
* * 
* *
* * Author:
* *   JEP, Josue Enrique Peraza Velazquez
* *
* *
*/
#pragma once
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <array>
#include "Log.h"
// ************************************************************************* //
// BCJR Decoder class: Uses the same CCSDS K=7 rate 1/2 inner code.
// Convention: coded LLR L = log P(bit=0)/P(bit=1) (L>0 favors bit 0)
// Free non-terminated endpoint trellis: start state is known (0), end free.
// ************************************************************************* //
class BCJRDecoder
{
public:
    // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Trellis structure
    // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    struct TrellisNode
    {
    private:
    uint8_t state{0};         // Current state
    uint8_t input{0};         // Input bit (0,1)
    uint8_t next[64][2];      // Next state for each input bit (0,1)
    uint8_t out[64][2][2];       // Output parity bits for each input bit (0,1)
    public:
    TrellisNode (void)=default;
    ~TrellisNode (void)=default;
    inline void Clear (void)
    {                         // ~~~~~~~~~~ Clear ~~~~~~~~~~ //
        for (int i=0;i<64;i++)  // For each state
        {                       //
        next[i][0]=0;         // Clear next state for input 0
        next[i][1]=0;         // Clear next state for input 1
        for (int j=0;j<2;j++)   // For each input bit
        {                     //
            out[i][j][0]=0;     // Clear output parity bit 1
            out[i][j][1]=0;     // Clear output parity bit 2
        }                     // Done for each input bit
        }                       //
    }                         // ~~~~~~~~~~ Clear ~~~~~~~~~~ //
    inline void Set (
        int state,              // Current state
        int input,              // Input bit (0,1)
        int nextstate,          // Next state
        int outbits)            // Output parity bits
    {                         // ~~~~~~~~~~ Set ~~~~~~~~~~ //
        next[state][input]=static_cast<uint8_t>(nextstate);// Set next state
        out[state][input][0]=static_cast<uint8_t>(outbits);  // Set output parity bits
    }                         // ~~~~~~~~~~ Set ~~~~~~~~~~ //
    inline void Get (
        int state,              // Current state
        int input,              // Input bit (0,1)
        int& nextstate,         // Next state (output)
        int& outbits) const     // Output parity bits (output)
    {                         // ~~~~~~~~~~ Get ~~~~~~~~~~ //
        nextstate=next[state][input];// Get next state
        outbits=out[state][input][0];  // Get output parity bits
    }                         // ~~~~~~~~~~ Get ~~~~~~~~~~ //
    inline uint8_t* GetNext (void)
    {
        return &this->next[0][0];
    }
    inline uint8_t* GetOut (void)
    {
        return &this->out[0][0][0];
    }
    inline uint8_t* GetNext (int state, int input)
    {
        return &this->next[state][input];
    }
    inline uint8_t* GetOut (int state, int input, int bit)
    {
        return &this->out[state][input][bit];
    }
    inline uint8_t GetCurrentState (void) const
    {
        return this->state;
    }
    inline void SetCurrentState (int state)
    {
        this->state=static_cast<uint8_t>(state);
    }
    inline uint8_t GetInputBit (void) const
    {
        return this->input;
    }
    inline void SetInputBit (int input)
    {
        this->input=static_cast<uint8_t>(input);
    }
    inline uint8_t Parity (uint32_t bs) // Bit sequence to parity check
    {                               // ~~~~~~~~~~ Parity ~~~~~~~~~~ //                                
        bs^=bs>>16u;                  // Fold upper bits
        bs^=bs>>8u;                   // Fold again
        bs^=bs>>4u;                   // Fold again
        bs^=bs>>2u;                   // Fold again
        bs^=bs>>1u;                   // Fold again
        return static_cast<uint8_t>(bs&0x1u);// Return parity bit
    }                               // ~~~~~~~~~~ Parity ~~~~~~~~~~ //
    inline void BuildTrellis (void)
    {                               // ~~~~~~~~~~~~ BuildTrellis ~~~~~~~~~~~~ //
        Clear();                      // Clear existing trellis
        const uint32_t g1=0b1111001u; // Generator 1
        const uint32_t g2=0b1011011u; // Generator 2
        for (uint8_t s=0;s<64;++s)    // For each state
        {                             // and for each input bit
        for (uint8_t b=0;b<2;++b)   // and for each input bit.....
        {                           // Compute Trellis diagram
            const int zprm=((s<<1)|b)&0x7F; // Shift register with input bit
            next[s][b]=static_cast<uint8_t>(zprm&0x3F);// Next state (6 LSBs)
            out[s][b][0]=static_cast<uint8_t>(Parity(static_cast<uint32_t>(zprm)&g1));// Expected C1 (G1)
            // ~~~~~~~~~~~~~~~~~~~~~~ //
            // CCSDS 131.0-B-5 sec 3.3.1(5): G2 output inversion. Mirrors the encoder
            // so the trellis expects the inverted C2 for rate 1/2 AND 3/4
            // (the puncture metric reads this same table entry)
            // ~~~~~~~~~~~~~~~~~~~~~~ //
            out[s][b][1]=static_cast<uint8_t>(Parity(static_cast<uint32_t>(zprm)&g2));// RAW
            // //Deb(" BuildTrellis: State %02X Input %d -> Next %02X Output [%d,%d]",
            //    s,b,next[s][b],out[s][b][0],out[s][b][1]);
        }                           // Done for each input bit
        }                             // Done for each state
    }                               // ~~~~~~~~~~~~ BuildTrellis ~~~~~~~~~~~~ //
    inline void Print (void) const
    {                               // ~~~~~~~~~~ Print ~~~~~~~~~~ //
        for (uint8_t s=0;s<64;++s)    // For each state
        {                             //
        for (uint8_t b=0;b<2;++b)   // For each input bit
        {                           //
            uint8_t ns=next[s][b];    // Next state
            uint8_t ob1=out[s][b][0]; // Output bit 1
            uint8_t ob2=out[s][b][1]; // Output bit 2
            //std::printf(" State %02X Input %d -> Next %02X Output [%d,%d]\n",
            //  s,b,ns,ob1,ob2);
            Log(" State %02X Input %d -> Next %02X Output [%d,%d]",
                s,b,ns,ob1,ob2);
        }                           // Done for each input bit
        }                             // Done for each state
    }                               // ~~~~~~~~~~ Print ~~~~~~~~~~ //        
    } trellis{};      

public:
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // BCJR Decoder LLR and decision struct
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    static constexpr float negeps=-1e30f; // Very negative LLR
    struct LLR
    {
      std::vector<float> lc1{};       // LLR of coded bit 1 per step (C1)
      std::vector<float> lc2{};       // LLR of coded bit 2 per step (C2)
      std::vector<uint8_t>* out{};    // Decision input bits (0/1 hard-bits)
      std::vector<float>* llr{};      // A-posteriori probability LLR for decoded bits (L>0 favors bit 0)
      inline void Clear (void)
      {
        lc1.clear();                  // Clear LLR for coded bit 1
        lc2.clear();                  // Clear LLR for coded bit 2
        if (out!=nullptr)             // We have output decoded bits?
        out->clear();               // Clear output decoded bits
        if (llr!=nullptr)             // We have output LLR for decoded bits?
        llr->clear();               // Clear output LLR for decoded bits
      }
    };
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Constructor: builds trellis structure
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    BCJRDecoder (void)
    {                                 // ~~~~~~~~~~ ctor ~~~~~~~~~~ //
      trellis.BuildTrellis();         // Build trellis diagram structure
    }                                 // ~~~~~~~~~~ ctor ~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // lc1,lc2: Channel LLRs for the coded bits of each info step.
    //          For punctured/erased symbols pass 0.0f (no information)
    //          C2 carries the CCSDS G2 inversion: flip sign of lc2 before calling,
    //          or pass it pre-flipped, to match the inverted-C2 trellis expectations.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    inline void Decode (const LLR& l)
    {
      if (l.out==nullptr||l.llr==nullptr) // No output decoding decisions, or APP LLR buffer?
        return;                       // Nothing to do
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Prepare output buffers
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      const int N=static_cast<int>(std::min(l.lc1.size(),l.lc2.size()));
      if (N==0) return;               // No input LLRs, nothing to do
      l.out->assign(N,0);             // Clear output decoded bits
      l.llr->assign(N,0.f);           // Clear output LLR for decoded bits
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // alpha[t][s], beta[t][s], t in [0..N]
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      std::vector<std::array<float,64>> a(N+1),b(N+1); // Forward and backward state metrics
      for (int t=0;t<=N;++t)          // For the number of steps....
      {                               // Init forward and backward metrics...
        a[t].fill(negeps);            // Init forward metrics to very negative
        b[t].fill(negeps);            // Init backward metrics to very negative
      }                               // Done initializing metrics
      a[0][0]=0.f;                    // Start state known, so forward metric is 0 for state 0
      for (int s=0;s<64;++s)          // Free end
        b[N][s]=0.f;                  // so backward metric is 0 for all states at end
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Gamma(s,b) at step t is defined by:
      //  0.5*(x1*lc1 + x2*lc2), x+=1 for bit 0, -1 for bit 1
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      auto gamma=[&](int t, int s, int bb)->float
      {
        int ns=0;                     // Next state for this transition
        int ob=0;                     // Output bits for this transition
        trellis.Get(s,bb,ns,ob);      // Get next state and output
        (void)ns;                     // Don't need it anymore (want to reuse trellis call)
        const uint8_t* e1=trellis.GetOut(s,bb,0); // Output bit 1 for this transition
        const uint8_t* e2=trellis.GetOut(s,bb,1); // Output bit 2 for this transition
        const float x1=(*e1==0)?1.f:-1.f; // Map expected bit output to +1 for bit 0, -1 for bit 1
        const float x2=(*e2==0)?1.f:-1.f; // Map expected bit output to +1 for bit 0, -1 for bit 1
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // A-priori on u assumed 0
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        return 0.5f*(x1*l.lc1[t]+x2*l.lc2[t]); // Return gamma metric for this transition
      };                              // Done gamma lambda function
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Forward recursion: compute alpha[t][s] for t in [1..N]
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      for (int t=0;t<N;++t)           // For each step
      {                               // (explicit braces: this block is its own pass)
        for (int s=0;s<64;++s)        // For each state
        {
          if (a[t][s]<=negeps)        // Invalid path?
              continue;               // Skip it
          for (int bb=0;bb<2;++bb)    // For each input bit
          {                           // compute alpha
            const int ns=static_cast<int>(trellis.GetNext(s,bb)[0]); // Next state for THIS transition
            const float m=a[t][s]+gamma(t,s,bb); // Metric for this transition
            if (m>a[t+1][ns])         // Better path to next state?
              a[t+1][ns]=m;           // Max-Log alpha update
          }                           // Done for each input bit
        }                             // Done for each state
      }                               // Done forward recursion
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Backward recursion: compute beta[t][s] for t in [N-1..0]
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      for (int t=N-1;t>=0;--t)        // For each step (backwards)
      {                               // (explicit braces: this block is its own pass)
        for (int s=0;s<64;++s)        // For each state
        {                             // compute beta
          for (int bb=0;bb<2;++bb)    // For each input bit
          {                           // compute beta
            const int ns=static_cast<int>(trellis.GetNext(s,bb)[0]); // Next state for THIS transition
            const float m=b[t+1][ns]+gamma(t,s,bb); // Metric for this transition
            if (m>b[t][s])            // Better path from this state?
              b[t][s]=m;              // Max-Log beta update
          }                           // Done for each input bit
        }                             // Done for each state
      }                               // Done backward recursion
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // A-Posteriori Probability per info bit
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      for (int t=0;t<N;++t)           // For each step
      {                               // Compute APP LLR for this step
        float m0=negeps;              // Metric for u=0
        float m1=negeps;              // Metric for u=1
        for (int s=0;s<64;++s)        // For each state
        {                             // Compute contribution to APP LLR for this state
          if (a[t][s]<=negeps)        // Invalid path?
            continue;                 // Skip it
          for (int bb=0;bb<2;++bb)    // For each input
          {                           // bit
            const int ns=static_cast<int>(trellis.GetNext(s,bb)[0]); // Next state for THIS transition
            const float m=a[t][s]+gamma(t,s,bb)+b[t+1][ns]; // Metric for this transition
            if (bb==0)                // Input bit 0?
            {                         // Better path?
              if (m>m0)               // Better path for u=0?
                m0=m;                 // Update metric for u=0
            }                         // Done for input bit 0
            else                      // Input bit 1?
            {                         // Better path?
              if (m>m1)               // Better path for u=1?
                m1=m;                 // Update metric for u=1
            }                         // Done for input bit 1
          }                           // Done for each input bit
        }                             // Done for each state
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Log P(u=0)/P(u=1) = log sum exp for u=0 - log sum exp for u=1
        // Max-Log approximation: log sum exp(x_i) ~ max(x_i)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        const float Lk=m0-m1;         // LLR for this bit
        (*l.llr)[static_cast<size_t>(t)]=Lk; // Store LLR for this bit
        (*l.out)[static_cast<size_t>(t)]=(Lk>=0.f)?0:1; // Make hard decision for this bit
      }                               // Done computing APP LLR and decisions for each bit
    }                                 // ~~~~~~~~~~ Decode ~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Build BCJR coded LLRs from soft symbols
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    void BuildCodedLLRs (
      const std::vector<float>& src,        // Soft symbols to build the LLRs
      std::vector<float>& lc1,              // Output LLRs for C1 (the only bit for rate 1/2 phase 1)
      std::vector<float>& lc2)              // Output LLRs for C2 (the only bit for rate 1/2 phase 2)
    {                                       // ~~~~~~~~ BuildCodedLLRs ~~~~~~~~~~~ //
      lc1.clear();                          // Clear any junk at that address
      lc2.clear();                          // Clear any junk at that address
      const size_t N=src.size();            // Number of coded soft symbols
      if (N==0u) return;                    // Nothing to estimate on or build
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // M2M4 blind SNR estimate over the entire coded rail 
      // (real anitpiodal stats) identical on C1/C2 and on I/Q so pooling minimizes
      // the estimator variance).
      // M2 = a^1 a s^2, M4=3*M2^2 - 2*a^4 -> S=A62, N=s^2, k=2a/s^2
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      double s2=0.0;                        // Second moment accumulator
      double s4=0.0;                        // Fourth moment accumulator
      for (float v:src)                     // For each soft symbol over the coded rail
      {                                     // Accumulate moments
        const double y2=static_cast<double>(v)*static_cast<double>(v); // Square of the symbol
        s2+=y2;                             // Accumulate second moment
        s4+=y2*y2;                          // Accumulate fourth moment
      }                                     // Done accumulating moments
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Computethe 2nd and 4th moment of E[y^2] and E[y^4] which is the expected
      // value of the second and fourth moment of the received symbols over the coded
      // rail.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      const double M2=s2/static_cast<double>(N); // 2nd moment E[y^2]
      const double M4=s4/static_cast<double>(N); // 4th moment E[y^4]
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Solve for signal power S=a^2 and noise power N = sigma^2, guarding for:
      // - radicand positivity (it cand go negative at low SNR or clipping) -> clamp to 0
      // - N = M2-S positivity such that S lies in [0,M2]
      // - Division by ~0 on a silent/clipped block, by flooring N 
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      const double rad=std::max(0.0,(3.0*M2*M2-M4)*0.5); // (2M2^2 - M4)/2 >= 0 radicand for the SNR estimate
      double S=std::sqrt(rad);              // Signal power estimate S=a^2
      S=std::clamp(S,0.0,M2);               // Keep N = M2 - S in [0,M2]
      const double N=std::max(M2-S,1e-6);   // Noise power (variance estimate) with a small floor
      const double a=std::sqrt(S);          // Amplitude estimate
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Channel reliability factor k = 1a/sigma^2 clamped to a sane band so a degenerate
      // block can never feed gamma a pathological scale. k only sets confidence
      // spread (not the hard decision boundary) so we clamp conservatively
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      const float k=std::clamp(static_cast<float>(2.0*a/N),0.1f,100.0f); // LLR spread factor
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Build per-info-step coded LLRs, mirroring the weighted Viterbi intake
      // where:
      // L2 -k*src (slicer maps s>=0 to bit 1; BCJR maps L>0 -> bit 0).
      // G2 inversion is Rate dependent: un-invert C2 (flip sign) for rate 1/2
      // per CCSDS 131.0-B. Punctured position carry lc=0 (no information)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      lc1.reserve(N);                       // Worst case sizing
      lc2.reserve(N);                       // Worst case sizing
      size_t idx=0u;                        // Stream cursor
      int ph=0;                             // Rate 3/4 puncture phase
      while (true)                          // Until we exhaust the stream...
      {                                     // Walk the coded rail
        float l1=0.f,l2=0.f;                // 0 ==erasure (no info)
        if (!cfg.vp34)                      // Rate 1/2?
        {                                   // Yes, both symbols in every step
          if (idx+1u>=N)                    // Enough symbols for another step?
            break;                          // No, done building LLRs
          l1=-k*src[idx++];                 // LLR for C1 (un-inverted)
          l2=+k*src[idx++];                 // LLR for C2 (flipped to undo CCSDS G2 inversion, rate 1/2 only)
        }                                   // Done rate 1/2
        else                                // Otherwise rate 3/4 phase machine
        {                                   // Mirror encoder
          if (ph==0)                        // Are we ate phase 0?
          {                                 // Yes C1,C2 both present
            if (idx+1u>=N)                  // Enough symbols for another step?
              break;                        // No, done building LLRs
            l1=-k*src[idx++];               // LLR for C1
            l2=-k*src[idx++];               // LLR for C2 (no flip for rate 3/4)
          }                                 // Done phase 0
          else if (ph==1)                   // Phase 1?
          {                                 // Yes, C2 present and C1 erased
            if (idx>=N)                     // Enough symbols for another step?
              break;                        // No, done building LLRs
            l2=-k*src[idx++];               // LLR for C2 (no flip)
          }                                 // Done phase 1
          else                              // Phase 2?
          {                                 // Yes, C1 present and C2 erased
            if (idx>=N)                     // Enough symbols for another step?
              break;                        // No, done building LLRs
            l1=-k*src[idx++];               // LLR for C1 (un-inverted)
          }                                 // Done phase 2
          ph=(ph+1)%3;                      // Advance the phase slice
        }                                   // Done rate 3/4
        lc1.push_back(l1);                  // Push LLR for C1 for this info
        lc2.push_back(l2);                  // Push LLR for C2 for this info
      }                                     // Done walking coded rail and building LLRs
    }                                       // ~~~~~~~~ BuildCodedLLRs ~~~~~~~~~~~ //
private:
    TrellisNode trellis{};
};      
